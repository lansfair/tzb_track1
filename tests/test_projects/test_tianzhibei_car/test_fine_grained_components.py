import cv2
import numpy as np
import torch

from projects.tianzhibei_car import (BalancedPrototypeContrastiveLoss,
                                     PrototypeDecoupledBBoxHead, aligned_crop,
                                     restricted_group_prediction)


def test_balanced_prototype_loss_backward():
    loss_module = BalancedPrototypeContrastiveLoss(
        num_classes=10, embedding_dim=16, loss_weight=0.05)
    features = torch.randn(12, 16, requires_grad=True)
    labels = torch.tensor([0] * 8 + [1, 1, 9, 10])
    loss = loss_module(features, labels, torch.ones(12))
    assert torch.isfinite(loss)
    loss.backward()
    assert features.grad is not None
    assert loss_module.prototypes.grad is not None


def test_prototype_head_preserves_bbox_path_and_adds_loss():
    head = PrototypeDecoupledBBoxHead(
        in_channels=256,
        fc_out_channels=128,
        roi_feat_size=7,
        num_classes=10,
        reg_class_agnostic=True,
        bbox_coder=dict(type='DeltaXYWHTRBBoxCoder', angle_version='le90'),
        loss_cls=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=False),
        loss_bbox=dict(type='mmdet.SmoothL1Loss', beta=1.0),
        embedding_dim=16,
        cls_branch_channels=32,
        loss_proto=dict(
            type='BalancedPrototypeContrastiveLoss',
            num_classes=10,
            embedding_dim=16,
            loss_weight=0.05))
    head.train()
    features = torch.randn(8, 256, 7, 7)
    cls_score, bbox_pred = head(features)
    labels = torch.tensor([0, 0, 1, 2, 5, 9, 10, 10])
    rois = torch.tensor([[0, 50, 50, 20, 10, 0.0]] * 8).float()
    bbox_targets = torch.zeros(8, 5)
    bbox_weights = torch.zeros(8, 5)
    bbox_weights[:6] = 1
    losses = head.loss(cls_score, bbox_pred, rois, labels, torch.ones(8),
                       bbox_targets, bbox_weights)
    assert 'loss_proto' in losses
    assert torch.isfinite(losses['loss_proto'])


def test_aligned_crop_and_restricted_relabelling():
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    points = cv2.boxPoints(((64, 64), (60, 20), 30)).astype(np.int32)
    cv2.fillConvexPoly(image, points, (255, 0, 0))
    crop = aligned_crop(image, (64, 64, 60, 20, np.deg2rad(30)), (128, 64))
    assert crop.shape == (64, 128, 3)
    assert crop[..., 0].mean() > 200

    probabilities = torch.tensor([
        [.8, .2, 0, 0, 0, 0, 0, 0, 0, 0],
        [.2, .8, 0, 0, 0, 0, 0, 0, 0, 0],
    ])
    labels, accepted = restricted_group_prediction(
        probabilities, torch.tensor([1, 0]), min_margin=0.1)
    assert labels.tolist() == [0, 1]
    assert accepted.tolist() == [True, True]
