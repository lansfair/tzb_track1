import torch
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
                int((weight > 0).sum())
                if weight is not None else loss.numel(), 1)
            loss = loss.sum() / denominator
        return loss * self.loss_weight


@MODELS.register_module()
class BalancedPrototypeContrastiveLoss(nn.Module):
    """Class-balanced prototype contrastive loss for foreground RoIs.

    The prototypes are learnable parameters and are therefore present even
    when a rare class is absent from the current mini-batch.  Losses are
    averaged within each present class before being averaged across classes,
    preventing the numerous Small Car/Van RoIs from dominating the auxiliary
    objective.  Background and zero-weight samples are ignored.
    """

    def __init__(self,
                 num_classes: int,
                 embedding_dim: int = 256,
                 temperature: float = 0.1,
                 loss_weight: float = 0.05) -> None:
        super().__init__()
        if num_classes <= 1:
            raise ValueError('num_classes must be greater than one')
        if temperature <= 0:
            raise ValueError('temperature must be positive')
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.loss_weight = loss_weight
        self.prototypes = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.normal_(self.prototypes, std=0.02)

    def forward(self, embeddings, labels, weights=None, **kwargs):
        if embeddings.ndim != 2 or embeddings.size(1) != self.embedding_dim:
            raise ValueError(
                f'Expected embeddings shaped [N, {self.embedding_dim}], '
                f'got {tuple(embeddings.shape)}')
        valid = (labels >= 0) & (labels < self.num_classes)
        if weights is not None:
            valid &= weights > 0
        if not valid.any():
            return embeddings.sum() * 0

        features = F.normalize(embeddings[valid].float(), dim=1)
        prototypes = F.normalize(self.prototypes.float(), dim=1)
        targets = labels[valid]
        logits = features @ prototypes.t() / self.temperature
        per_sample = F.cross_entropy(logits, targets, reduction='none')
        if weights is not None:
            sample_weights = weights[valid].float()
        else:
            sample_weights = per_sample.new_ones(per_sample.shape)

        class_losses = []
        for class_id in targets.unique(sorted=True):
            class_mask = targets == class_id
            denominator = sample_weights[class_mask].sum().clamp_min(1e-6)
            class_losses.append(
                (per_sample[class_mask] * sample_weights[class_mask]).sum() /
                denominator)
        return torch.stack(class_losses).mean() * self.loss_weight
