import torch
import torch.nn as nn


class DualCropConvNeXtTiny(nn.Module):
    """Shared ConvNeXt-T encoder for tight and 2x-context vehicle crops."""

    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        super().__init__()
        from torchvision.models import (ConvNeXt_Tiny_Weights,
                                        convnext_tiny)
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
            [self.encode(tight_crop), self.encode(context_crop)], dim=1)
        return self.classifier(features)


def geometric_probability_fusion(detector_probability,
                                 classifier_probability,
                                 detector_weight: float = 0.6):
    """Fuse calibrated detector/refiner probabilities geometrically."""
    fused = (detector_probability.clamp_min(1e-8)**detector_weight *
             classifier_probability.clamp_min(1e-8)**(1-detector_weight))
    return fused / fused.sum(dim=-1, keepdim=True)
