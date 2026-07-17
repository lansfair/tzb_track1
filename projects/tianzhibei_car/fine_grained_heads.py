"""Fine-grained RoI heads for the Tianzhibei vehicle experiments."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.structures.bbox import get_box_tensor
from mmcv.ops import box_iou_rotated
from torch import Tensor

from mmrotate.models.roi_heads.bbox_heads import RotatedShared2FCBBoxHead
from mmrotate.registry import MODELS


@MODELS.register_module()
class PrototypeDecoupledBBoxHead(RotatedShared2FCBBoxHead):
    """Shared2FC head with a lightweight classification-only residual path.

    The original Shared2FC feature remains the sole input of bbox regression,
    so an existing checkpoint loads all baseline layers unchanged.  A compact
    Conv-GAP branch modifies only classification features.  Its projection
    head and prototype loss are active during training and add no inference
    cost.
    """

    def __init__(self,
                 cls_branch_channels: int = 256,
                 embedding_dim: int = 256,
                 cls_residual_scale: float = 0.1,
                 loss_proto: Optional[dict] = None,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        num_groups = 32 if cls_branch_channels % 32 == 0 else 1
        self.cls_residual = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                cls_branch_channels,
                kernel_size=3,
                padding=1,
                bias=False), nn.GroupNorm(num_groups, cls_branch_channels),
            nn.GELU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(cls_branch_channels, self.fc_out_channels), nn.GELU())
        self.cls_residual_scale = nn.Parameter(
            torch.tensor(float(cls_residual_scale)))
        self.proto_projector = nn.Sequential(
            nn.Linear(self.fc_out_channels, self.fc_out_channels), nn.GELU(),
            nn.Linear(self.fc_out_channels, embedding_dim))
        if loss_proto is None:
            loss_proto = dict(
                type='BalancedPrototypeContrastiveLoss',
                num_classes=self.num_classes,
                embedding_dim=embedding_dim,
                temperature=0.1,
                loss_weight=0.05)
        else:
            loss_proto = loss_proto.copy()
            loss_proto.setdefault('num_classes', self.num_classes)
            loss_proto.setdefault('embedding_dim', embedding_dim)
        self.loss_proto = MODELS.build(loss_proto)
        self._proto_features = None

    def init_weights(self) -> None:
        super().init_weights()
        # Make the first forward exactly match the loaded baseline classifier.
        # The residual branch then grows smoothly from zero during fine-tuning.
        residual_projection = self.cls_residual[-2]
        nn.init.zeros_(residual_projection.weight)
        nn.init.zeros_(residual_projection.bias)

    def forward(self, x: Tensor) -> tuple:
        roi_features = x
        for conv in self.shared_convs:
            x = conv(x)
        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.flatten(1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))

        x_reg = x
        cls_residual = self.cls_residual(roi_features)
        x_cls = x + self.cls_residual_scale * cls_residual
        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        self._proto_features = (
            self.proto_projector(x_cls) if self.training else None)
        return cls_score, bbox_pred

    def loss(self,
             cls_score: Tensor,
             bbox_pred: Tensor,
             rois: Tensor,
             labels: Tensor,
             label_weights: Tensor,
             bbox_targets: Tensor,
             bbox_weights: Tensor,
             reduction_override: Optional[str] = None) -> dict:
        losses = super().loss(
            cls_score,
            bbox_pred,
            rois,
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            reduction_override=reduction_override)
        proto_features = self._proto_features
        self._proto_features = None
        if proto_features is not None:
            losses['loss_proto'] = self.loss_proto(proto_features, labels,
                                                   label_weights)
        return losses


@MODELS.register_module()
class MildAdaptiveRecognitionBBoxHead(RotatedShared2FCBBoxHead):
    """Shared2FC head with detached, bounded proposal-quality weighting.

    Positive classification weights combine foreground confidence and refined
    rotated IoU.  A non-zero floor preserves difficult tiny vehicles, while
    mean normalization and an upper bound prevent a few easy samples from
    dominating a batch.
    """

    def __init__(self,
                 arl_gamma: float = 1.0,
                 arl_beta: float = 1.0,
                 arl_min_weight: float = 0.25,
                 arl_max_weight: float = 2.0,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not 0 <= arl_min_weight <= 1:
            raise ValueError('arl_min_weight must be in [0, 1]')
        self.arl_gamma = arl_gamma
        self.arl_beta = arl_beta
        self.arl_min_weight = arl_min_weight
        self.arl_max_weight = arl_max_weight

    @torch.no_grad()
    def _quality_weights(self, cls_score, bbox_pred, rois, labels,
                         bbox_targets):
        positive = (labels >= 0) & (labels < self.num_classes)
        if not positive.any():
            return positive, None, None
        positive_labels = labels[positive]
        confidence = F.softmax(
            cls_score.float(), dim=-1)[positive, positive_labels]

        if self.reg_class_agnostic:
            positive_deltas = bbox_pred.reshape(bbox_pred.size(0),
                                                -1)[positive]
        else:
            positive_deltas = bbox_pred.reshape(
                bbox_pred.size(0), self.num_classes, -1)[positive,
                                                         positive_labels]
        positive_rois = rois[positive, 1:]
        predicted = get_box_tensor(
            self.bbox_coder.decode(positive_rois, positive_deltas))
        target = get_box_tensor(
            self.bbox_coder.decode(positive_rois, bbox_targets[positive]))
        iou = box_iou_rotated(
            predicted.float(), target.float(), aligned=True).clamp(0, 1)
        raw = confidence.pow(self.arl_gamma) * iou.pow(self.arl_beta)
        quality = self.arl_min_weight + (1 - self.arl_min_weight) * raw
        quality = quality / quality.mean().clamp_min(1e-6)
        quality = quality.clamp(max=self.arl_max_weight)
        return positive, quality.to(cls_score.dtype), iou

    def loss(self,
             cls_score: Tensor,
             bbox_pred: Tensor,
             rois: Tensor,
             labels: Tensor,
             label_weights: Tensor,
             bbox_targets: Tensor,
             bbox_weights: Tensor,
             reduction_override: Optional[str] = None) -> dict:
        adjusted_weights = label_weights.clone()
        positive, quality, iou = self._quality_weights(cls_score, bbox_pred,
                                                       rois, labels,
                                                       bbox_targets)
        if quality is not None:
            adjusted_weights[positive] *= quality
        losses = super().loss(
            cls_score,
            bbox_pred,
            rois,
            labels,
            adjusted_weights,
            bbox_targets,
            bbox_weights,
            reduction_override=reduction_override)
        if quality is not None:
            losses['arl_mean_weight'] = quality.mean()
            losses['arl_mean_iou'] = iou.mean()
        return losses
