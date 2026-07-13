import torch.nn as nn
import torch.nn.functional as F

from mmrotate.registry import MODELS


@MODELS.register_module()
class LabelSmoothCrossEntropyLoss(nn.Module):
    """Cross entropy with label smoothing for the RoI classifier."""

    custom_cls_channels = False
    custom_activation = False

    def __init__(self,
                 label_smoothing: float = 0.05,
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        reduction = reduction_override or 'none'
        loss = F.cross_entropy(
            pred,
            label,
            reduction=reduction,
            label_smoothing=self.label_smoothing)
        if weight is not None:
            loss = loss * weight
        if reduction == 'none':
            denominator = avg_factor if avg_factor is not None else max(
                int((weight > 0).sum()) if weight is not None else loss.numel(),
                1)
            loss = loss.sum() / denominator
        return loss * self.loss_weight
