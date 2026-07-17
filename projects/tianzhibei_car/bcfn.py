"""Adjacent-level bilinear RoI fusion for fine-grained vehicles."""

import torch
import torch.nn as nn

from mmrotate.models.roi_heads.roi_extractors import RotatedSingleRoIExtractor
from mmrotate.registry import MODELS


@MODELS.register_module()
class AdjacentLevelBilinearRoIExtractor(RotatedSingleRoIExtractor):
    """Fuse the assigned FPN level with one adjacent semantic level.

    Tiny P2 vehicles are paired with P3, giving the classifier additional
    context without removing the stride-4 detail.  Two low-rank projections
    interact multiplicatively to form a gate, while a learnable small residual
    scale keeps initialization close to the baseline extractor.
    """

    def __init__(self,
                 *args,
                 bilinear_channels: int = 64,
                 fusion_scale: float = 0.1,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.primary_reduce = nn.Conv2d(self.out_channels, bilinear_channels,
                                        1)
        self.adjacent_reduce = nn.Conv2d(self.out_channels, bilinear_channels,
                                         1)
        self.gate_expand = nn.Sequential(
            nn.Conv2d(bilinear_channels, self.out_channels, 1), nn.Sigmoid())
        self.adjacent_transform = nn.Conv2d(
            self.out_channels, self.out_channels, 1, bias=False)
        self.fusion_scale = nn.Parameter(torch.tensor(float(fusion_scale)))

    def init_weights(self) -> None:
        super().init_weights()
        # Preserve the baseline RoI tensor at step zero.  The adjacent-level
        # residual is learned without injecting a random feature shift.
        nn.init.zeros_(self.adjacent_transform.weight)

    @staticmethod
    def _adjacent_level(level: int, num_levels: int) -> int:
        if num_levels == 1:
            return 0
        return level + 1 if level + 1 < num_levels else level - 1

    def _fuse(self, primary, adjacent):
        interaction = (
            self.primary_reduce(primary) * self.adjacent_reduce(adjacent))
        gate = self.gate_expand(interaction)
        residual = gate * self.adjacent_transform(adjacent)
        return primary + self.fusion_scale * residual

    def forward(self, feats, rois, roi_scale_factor=None):
        rois = rois.type_as(feats[0])
        out_size = self.roi_layers[0].output_size
        num_levels = len(feats)
        roi_feats = feats[0].new_zeros(
            rois.size(0), self.out_channels, *out_size)
        if rois.numel() == 0:
            return roi_feats + sum(p.sum() for p in self.parameters()) * 0
        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)
        target_levels = self.map_roi_levels(rois, num_levels)

        for level in range(num_levels):
            indices = (target_levels == level).nonzero(
                as_tuple=False).squeeze(1)
            if indices.numel() > 0:
                selected_rois = rois[indices]
                adjacent_level = self._adjacent_level(level, num_levels)
                primary = self.roi_layers[level](feats[level], selected_rois)
                adjacent = self.roi_layers[adjacent_level](
                    feats[adjacent_level], selected_rois)
                roi_feats[indices] = self._fuse(primary, adjacent)
            else:
                roi_feats += feats[level].sum() * 0
        return roi_feats
