"""Direction-aligned RGB crop classifier and supporting utilities."""

import csv
import math
import random
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

VEHICLE_CLASSES = ('Small Car', 'Van', 'Dump Truck', 'Cargo Truck',
                   'other-vehicle', 'Bus', 'Truck Tractor', 'Excavator',
                   'Trailer', 'Tractor')

AMBIGUOUS_CLASS_GROUPS = ((0, 1), (2, 3), (6, 8), (7, 9))


def read_geotiff_rgb(path):
    """Read the first three TIFF bands as contiguous RGB using GDAL."""
    try:
        from osgeo import gdal
    except ImportError as exc:
        raise ImportError('GDAL Python bindings are required to read TIFFs') \
            from exc
    dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
    if dataset is None or dataset.RasterCount < 3:
        raise ValueError(f'Cannot read three RGB bands from {path}')
    image = dataset.ReadAsArray(band_list=[1, 2, 3])
    dataset = None
    if image is None or image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f'Invalid RGB array in {path}')
    return np.ascontiguousarray(np.moveaxis(image, 0, -1))


def qbox_to_rbox(points: np.ndarray) -> np.ndarray:
    """Convert four polygon points to ``cx, cy, long, short, angle(rad)``."""
    import cv2
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    (cx, cy), (width, height), angle_deg = cv2.minAreaRect(points)
    if height > width:
        width, height = height, width
        angle_deg += 90.0
    angle = math.radians(angle_deg)
    angle = (angle + math.pi / 2) % math.pi - math.pi / 2
    return np.asarray([cx, cy, width, height, angle], dtype=np.float32)


def _rbox_corners(rbox: Sequence[float], scale: float = 1.0):
    cx, cy, width, height, angle = map(float, rbox)
    if height > width:
        width, height = height, width
        angle += math.pi / 2
    half_width = width * scale / 2
    half_height = height * scale / 2
    axis_long = np.asarray([math.cos(angle), math.sin(angle)], np.float32)
    axis_short = np.asarray([-math.sin(angle), math.cos(angle)], np.float32)
    center = np.asarray([cx, cy], np.float32)
    return np.stack([
        center - half_width * axis_long - half_height * axis_short,
        center + half_width * axis_long - half_height * axis_short,
        center + half_width * axis_long + half_height * axis_short,
        center - half_width * axis_long + half_height * axis_short,
    ]).astype(np.float32)


def aligned_crop(image_rgb: np.ndarray,
                 rbox: Sequence[float],
                 output_size: Tuple[int, int],
                 scale: float = 1.0) -> np.ndarray:
    """Warp a rotated box so its long axis is horizontal.

    Args:
        image_rgb: HWC RGB uint8 image.
        rbox: ``cx, cy, width, height, angle`` with angle in radians.
        output_size: ``(width, height)`` of the returned crop.
        scale: Multiplicative context expansion around the box center.
    """
    import cv2
    output_width, output_height = output_size
    source = _rbox_corners(rbox, scale=scale)
    destination = np.asarray(
        [[0, 0], [output_width - 1, 0], [output_width - 1, output_height - 1],
         [0, output_height - 1]],
        dtype=np.float32)
    transform = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(
        image_rgb,
        transform, (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101)


def extract_dual_aligned_crops(image_rgb: np.ndarray,
                               rbox: Sequence[float],
                               tight_size=(128, 64),
                               context_size=(192, 96),
                               context_scale: float = 1.2):
    return (aligned_crop(image_rgb, rbox, tight_size, 1.0),
            aligned_crop(image_rgb, rbox, context_size, context_scale))


class AlignedVehicleCropDataset(Dataset):
    """CSV-backed paired tight/context crop dataset."""

    def __init__(
        self,
        manifest,
        training: bool = False,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ) -> None:
        self.manifest = Path(manifest)
        with self.manifest.open('r', encoding='utf-8-sig', newline='') as f:
            self.rows = list(csv.DictReader(f))
        if not self.rows:
            raise ValueError(f'Empty crop manifest: {manifest}')
        self.training = training
        self.mean = torch.tensor(mean, dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(std, dtype=torch.float32)[:, None, None]
        self.labels = [int(row['label']) for row in self.rows]

    def __len__(self):
        return len(self.rows)

    @staticmethod
    def _read_rgb(path):
        import cv2
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        return np.ascontiguousarray(image[..., ::-1])

    def _paired_transform(self, tight, context):
        from torchvision.transforms import functional as TF
        tight = TF.to_tensor(tight)
        context = TF.to_tensor(context)
        if self.training:
            if random.random() < 0.5:
                tight, context = TF.hflip(tight), TF.hflip(context)
            if random.random() < 0.5:
                tight, context = TF.vflip(tight), TF.vflip(context)
            brightness = random.uniform(0.85, 1.15)
            contrast = random.uniform(0.85, 1.15)
            saturation = random.uniform(0.9, 1.1)
            for operation, factor in ((TF.adjust_brightness, brightness),
                                      (TF.adjust_contrast, contrast),
                                      (TF.adjust_saturation, saturation)):
                tight, context = operation(tight,
                                           factor), operation(context, factor)
        return ((tight - self.mean) / self.std,
                (context - self.mean) / self.std)

    def __getitem__(self, index):
        row = self.rows[index]
        tight = self._read_rgb(self.manifest.parent / row['tight_path'])
        context = self._read_rgb(self.manifest.parent / row['context_path'])
        tight, context = self._paired_transform(tight, context)
        return tight, context, int(row['label']), row['image_id']


class DualCropConvNeXtTiny(nn.Module):
    """Shared ConvNeXt-T encoder for tight and context vehicle crops."""

    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        super().__init__()
        from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        network = convnext_tiny(weights=weights)
        self.features = network.features
        self.pool = network.avgpool
        feature_dim = network.classifier[-1].in_features
        self.classifier = nn.Sequential(
            nn.LayerNorm(2 * feature_dim),
            nn.Linear(2 * feature_dim, num_classes))

    def encode(self, image):
        return torch.flatten(self.pool(self.features(image)), 1)

    def forward(self, tight_crop, context_crop):
        features = torch.cat(
            [self.encode(tight_crop),
             self.encode(context_crop)], dim=1)
        return self.classifier(features)


def geometric_probability_fusion(detector_probability,
                                 classifier_probability,
                                 detector_weight: float = 0.6):
    """Fuse calibrated detector/refiner probabilities geometrically."""
    fused = (
        detector_probability.clamp_min(1e-8)**detector_weight *
        classifier_probability.clamp_min(1e-8)**(1 - detector_weight))
    return fused / fused.sum(dim=-1, keepdim=True)


def restricted_group_prediction(probabilities: torch.Tensor,
                                detector_labels: torch.Tensor,
                                groups=AMBIGUOUS_CLASS_GROUPS,
                                min_confidence: float = 0.6,
                                min_margin: float = 0.15):
    """Relabel only confident predictions within predefined confusion pairs."""
    output_labels = detector_labels.clone()
    accepted = torch.zeros_like(detector_labels, dtype=torch.bool)
    for group in groups:
        group_tensor = detector_labels.new_tensor(group)
        candidates = (detector_labels[:, None] == group_tensor[None]).any(1)
        if not candidates.any():
            continue
        group_probs = probabilities[candidates][:, group_tensor]
        values, indices = group_probs.sort(dim=1, descending=True)
        confident = ((values[:, 0] >= min_confidence) &
                     ((values[:, 0] - values[:, 1]) >= min_margin))
        candidate_indices = candidates.nonzero(as_tuple=False).squeeze(1)
        selected = candidate_indices[confident]
        output_labels[selected] = group_tensor[indices[confident, 0]]
        accepted[selected] = True
    return output_labels, accepted
