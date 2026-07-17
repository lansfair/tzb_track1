data_root = '/mnt/ht2-nas2/EO_test/wyf/tzb/data/car_det_train'

angle_version = 'le90'
auto_scale_lr = dict(base_batch_size=16, enable=False)
custom_hooks = [
    dict(type='mmdet.NumClassCheckHook'),
    dict(
        ema_type='mmdet.ExpMomentumEMA',
        momentum=0.0002,
        priority=49,
        type='EMAHook',
        update_buffers=True),
    dict(
        switch_epoch=28,
        switch_pipeline=[
            dict(alpha_policy='ignore', type='LoadGeoTiffRGB'),
            dict(
                box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
            dict(
                box_type_mapping=dict(gt_bboxes='rbox'),
                type='ConvertBoxType'),
            dict(size=1024, type='ResizeAndPad'),
            dict(angles=[
                90,
                180,
                270,
            ], prob=0.5, type='RandomChoiceRotate'),
            dict(angle_range=15, prob=0.2, type='RandomRotate'),
            dict(
                direction=[
                    'horizontal',
                    'vertical',
                    'diagonal',
                ],
                prob=0.75,
                type='mmdet.RandomFlip'),
            dict(type='mmdet.PhotoMetricDistortion'),
            dict(
                prob=0.1, sigma_range=(
                    1.0,
                    4.0,
                ), type='RandomGaussianNoise'),
            dict(
                prob=0.05, sigma_range=(
                    0.1,
                    0.8,
                ), type='RandomGaussianBlur'),
            dict(type='mmdet.PackDetInputs'),
        ],
        type='mmdet.PipelineSwitchHook'),
    dict(
        switch_epoch=36,
        switch_pipeline=[
            dict(alpha_policy='ignore', type='LoadGeoTiffRGB'),
            dict(
                box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
            dict(
                box_type_mapping=dict(gt_bboxes='rbox'),
                type='ConvertBoxType'),
            dict(size=1024, type='ResizeAndPad'),
            dict(
                direction=[
                    'horizontal',
                    'vertical',
                ],
                prob=0.5,
                type='mmdet.RandomFlip'),
            dict(type='mmdet.PackDetInputs'),
        ],
        type='mmdet.PipelineSwitchHook'),
    dict(
        stages=[
            dict(begin_epoch=0, name='A-main'),
            dict(begin_epoch=28, lr=2e-05, name='B-tail'),
            dict(begin_epoch=36, lr=1e-05, name='C-calibrate'),
        ],
        type='TianzhibeiStageHook'),
]
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'projects.tianzhibei_car',
        'projects.tianzhibei_car.mtp_backbone',
    ])
dataset_common = dict(
    boundary_mode='refit',
    data_prefix=dict(ann_path='gt/', img_path='input_path/'),
    data_root=data_root,
    drop_invalid=True,
    filter_cfg=dict(filter_empty_gt=True),
    img_suffix='.tif',
    min_box_area=1.0,
    min_box_side=1.0,
    min_visible_ratio=0.5,
    type='TianzhibeiCarDataset')
default_hooks = dict(
    checkpoint=dict(
        interval=2,
        max_keep_ckpts=3,
        save_best='auto',
        save_last=True,
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))
default_scope = 'mmrotate'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
formal_val_pipeline = [
    dict(alpha_policy='ignore', type='LoadGeoTiffRGB'),
    dict(box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
    dict(box_type_mapping=dict(gt_bboxes='rbox'), type='ConvertBoxType'),
    dict(size=1024, type='ResizeAndPad'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'geo_transform',
            'projection',
        ),
        type='mmdet.PackDetInputs'),
]
load_from = '/mnt/ht2-nas2/EO_test/tianzhibei/weights/mtp-vit-l-rvsa-fair1m-to-tianzhibei-1024.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        attn_drop_rate=0.0,
        depth=24,
        drop_path_rate=0.3,
        drop_rate=0.0,
        embed_dim=1024,
        img_size=1024,
        interval=6,
        mlp_ratio=4,
        num_classes=10,
        num_heads=16,
        out_indices=[
            7,
            11,
            15,
            23,
        ],
        patch_size=16,
        pretrained=None,
        qk_scale=None,
        qkv_bias=True,
        type='RVSA_MTP_branches',
        use_abs_pos_emb=True,
        use_checkpoint=False),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        boxtype2tensor=False,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='mmdet.DetDataPreprocessor'),
    neck=dict(
        in_channels=[
            1024,
            1024,
            1024,
            1024,
        ],
        num_outs=5,
        out_channels=256,
        type='mmdet.FPN'),
    roi_head=dict(
        bbox_head=dict(
            bbox_coder=dict(
                angle_version='le90',
                edge_swap=True,
                norm_factor=None,
                proj_xy=True,
                target_means=(
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ),
                target_stds=(
                    0.1,
                    0.1,
                    0.2,
                    0.2,
                    0.1,
                ),
                type='DeltaXYWHTRBBoxCoder'),
            cls_predictor_cfg=dict(type='mmdet.Linear'),
            fc_out_channels=1024,
            in_channels=256,
            loss_bbox=dict(type='KFLoss', fun='ln', loss_weight=5.0),
            loss_bbox_type='kfiou',
            loss_cls=dict(
                label_smoothing=0.05,
                loss_weight=1.0,
                type='LabelSmoothCrossEntropyLoss'),
            num_classes=10,
            predict_box_type='rbox',
            reg_class_agnostic=True,
            reg_predictor_cfg=dict(type='mmdet.Linear'),
            roi_feat_size=7,
            type='RotatedShared2FCBBoxHead'),
        bbox_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(
                clockwise=True,
                out_size=7,
                sample_num=2,
                type='RoIAlignRotated'),
            type='RotatedSingleRoIExtractor'),
        type='mmdet.StandardRoIHead'),
    rpn_head=dict(
        anchor_generator=dict(
            ratios=[
                0.25,
                0.5,
                1.0,
                2.0,
                4.0,
            ],
            scales=[
                4,
            ],
            strides=[
                4,
                8,
                16,
                32,
                64,
            ],
            type='mmdet.AnchorGenerator',
            use_box_type=True),
        bbox_coder=dict(
            angle_version='le90',
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
                0.5,
                0.5,
            ],
            type='MidpointOffsetCoder'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(
            beta=0.1111111111111111,
            loss_weight=1.0,
            type='mmdet.SmoothL1Loss'),
        loss_cls=dict(
            loss_weight=1.0, type='mmdet.CrossEntropyLoss', use_sigmoid=True),
        type='OrientedRPNHead'),
    test_cfg=dict(
        rcnn=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.1, type='nms_rotated'),
            nms_pre=2000,
            score_thr=0.01),
        rpn=dict(
            max_per_img=2000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.8, type='nms'),
            nms_pre=8000)),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                ignore_iof_thr=-1,
                iou_calculator=dict(type='RBboxOverlaps2D'),
                match_low_quality=False,
                min_pos_iou=0.5,
                neg_iou_thr=0.5,
                pos_iou_thr=0.5,
                type='mmdet.MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=True,
                neg_pos_ub=-1,
                num=512,
                pos_fraction=0.25,
                type='mmdet.RandomSampler')),
        rpn=dict(
            allowed_border=-1,
            assigner=dict(
                ignore_iof_thr=-1,
                iou_calculator=dict(type='RBbox2HBboxOverlaps2D'),
                match_low_quality=True,
                min_pos_iou=0.3,
                neg_iou_thr=0.3,
                pos_iou_thr=0.5,
                type='mmdet.MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=False,
                neg_pos_ub=-1,
                num=512,
                pos_fraction=0.5,
                type='mmdet.RandomSampler')),
        rpn_proposal=dict(
            max_per_img=2000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.8, type='nms'),
            nms_pre=8000)),
    type='mmdet.FasterRCNN')
model_wrapper_cfg = dict(
    bucket_cap_mb=50,
    find_unused_parameters=False,
    gradient_as_bucket_view=True,
    type='MMDistributedDataParallel')
num_classes = 10
optim_wrapper = dict(
    accumulative_counts=1,
    clip_grad=dict(max_norm=35, norm_type=2),
    loss_scale=dict(growth_interval=1000, init_scale=512.0),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        fused=True,
        lr=0.0001,
        type='AdamW',
        weight_decay=0.05),
    paramwise_cfg=dict(
        bias_decay_mult=0.0, bypass_duplicate=True, norm_decay_mult=0.0),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=63,
        start_factor=0.3333333333333333,
        type='LinearLR'),
    dict(
        T_max=28,
        begin=0,
        by_epoch=True,
        end=28,
        eta_min=4e-05,
        type='CosineAnnealingLR'),
]
resume = False
stage_a_pipeline = [
    dict(alpha_policy='ignore', type='LoadGeoTiffRGB'),
    dict(box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
    dict(box_type_mapping=dict(gt_bboxes='rbox'), type='ConvertBoxType'),
    dict(size=1024, type='ResizeAndPad'),
    dict(angles=[
        90,
        180,
        270,
    ], prob=0.5, type='RandomChoiceRotate'),
    dict(
        direction=[
            'horizontal',
            'vertical',
            'diagonal',
        ],
        prob=0.75,
        type='mmdet.RandomFlip'),
    dict(type='mmdet.PhotoMetricDistortion'),
    dict(type='mmdet.PackDetInputs'),
]
stage_b_pipeline = [
    dict(alpha_policy='ignore', type='LoadGeoTiffRGB'),
    dict(box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
    dict(box_type_mapping=dict(gt_bboxes='rbox'), type='ConvertBoxType'),
    dict(size=1024, type='ResizeAndPad'),
    dict(angles=[
        90,
        180,
        270,
    ], prob=0.5, type='RandomChoiceRotate'),
    dict(angle_range=15, prob=0.2, type='RandomRotate'),
    dict(
        direction=[
            'horizontal',
            'vertical',
            'diagonal',
        ],
        prob=0.75,
        type='mmdet.RandomFlip'),
    dict(type='mmdet.PhotoMetricDistortion'),
    dict(prob=0.1, sigma_range=(
        1.0,
        4.0,
    ), type='RandomGaussianNoise'),
    dict(prob=0.05, sigma_range=(
        0.1,
        0.8,
    ), type='RandomGaussianBlur'),
    dict(type='mmdet.PackDetInputs'),
]
stage_c_pipeline = [
    dict(alpha_policy='ignore', type='LoadGeoTiffRGB'),
    dict(box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
    dict(box_type_mapping=dict(gt_bboxes='rbox'), type='ConvertBoxType'),
    dict(size=1024, type='ResizeAndPad'),
    dict(
        direction=[
            'horizontal',
            'vertical',
        ],
        prob=0.5,
        type='mmdet.RandomFlip'),
    dict(type='mmdet.PackDetInputs'),
]
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file='splits/val.txt',
        boundary_mode='refit',
        data_prefix=dict(ann_path='gt/', img_path='input_path/'),
        data_root=data_root,
        drop_invalid=True,
        filter_cfg=dict(filter_empty_gt=True),
        img_suffix='.tif',
        min_box_area=1.0,
        min_box_side=1.0,
        min_visible_ratio=0.5,
        pipeline=[
            dict(alpha_policy='ignore', type='LoadGeoTiffRGB'),
            dict(
                box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
            dict(
                box_type_mapping=dict(gt_bboxes='rbox'),
                type='ConvertBoxType'),
            dict(size=1024, type='ResizeAndPad'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'geo_transform',
                    'projection',
                ),
                type='mmdet.PackDetInputs'),
        ],
        test_mode=True,
        type='TianzhibeiCarDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(metric='mAP', type='DOTAMetric')
train_cfg = dict(
    dynamic_intervals=[
        (
            5,
            2,
        ),
        (
            28,
            1,
        ),
    ],
    max_epochs=40,
    type='EpochBasedTrainLoop',
    val_interval=1)
train_dataloader = dict(
    batch_sampler=None,
    batch_size=4,
    dataset=dict(
        ann_file='splits/train.txt',
        boundary_mode='refit',
        data_prefix=dict(ann_path='gt/', img_path='input_path/'),
        data_root=data_root,
        drop_invalid=True,
        filter_cfg=dict(filter_empty_gt=True),
        img_suffix='.tif',
        min_box_area=1.0,
        min_box_side=1.0,
        min_visible_ratio=0.5,
        pipeline=[
            dict(alpha_policy='ignore', type='LoadGeoTiffRGB'),
            dict(
                box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
            dict(
                box_type_mapping=dict(gt_bboxes='rbox'),
                type='ConvertBoxType'),
            dict(size=1024, type='ResizeAndPad'),
            dict(angles=[
                90,
                180,
                270,
            ], prob=0.5, type='RandomChoiceRotate'),
            dict(
                direction=[
                    'horizontal',
                    'vertical',
                    'diagonal',
                ],
                prob=0.75,
                type='mmdet.RandomFlip'),
            dict(type='mmdet.PhotoMetricDistortion'),
            dict(type='mmdet.PackDetInputs'),
        ],
        type='TianzhibeiCarDataset'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(
        dense_threshold=118,
        hard_fraction=0.2,
        ordinary_fraction=0.5,
        phase_schedule=[
            dict(
                begin_epoch=28,
                hard_fraction=0.2,
                hard_ids_file='work_dirs/tianzhibei/hard_train_ids.txt',
                ordinary_fraction=0.3,
                rare_fraction=0.5),
            dict(
                begin_epoch=36,
                hard_fraction=0.0,
                hard_ids_file=None,
                ordinary_fraction=1.0,
                rare_fraction=0.0),
        ],
        rare_fraction=0.3,
        rare_labels=(
            5,
            6,
            7,
            8,
            9,
        ),
        type='TianzhibeiBalancedSampler'))
train_pipeline = [
    dict(alpha_policy='warn', type='LoadGeoTiffRGB'),
    dict(box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
    dict(box_type_mapping=dict(gt_bboxes='rbox'), type='ConvertBoxType'),
    dict(size=1024, type='ResizeAndPad'),
    dict(
        direction=[
            'horizontal',
            'vertical',
            'diagonal',
        ],
        prob=0.75,
        type='mmdet.RandomFlip'),
    dict(type='mmdet.PhotoMetricDistortion'),
    dict(type='mmdet.PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file='splits/val.txt',
        boundary_mode='refit',
        data_prefix=dict(ann_path='gt/', img_path='input_path/'),
        data_root=data_root,
        drop_invalid=True,
        filter_cfg=dict(filter_empty_gt=True),
        img_suffix='.tif',
        min_box_area=1.0,
        min_box_side=1.0,
        min_visible_ratio=0.5,
        pipeline=[
            dict(alpha_policy='ignore', type='LoadGeoTiffRGB'),
            dict(
                box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
            dict(
                box_type_mapping=dict(gt_bboxes='rbox'),
                type='ConvertBoxType'),
            dict(size=1024, type='ResizeAndPad'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'geo_transform',
                    'projection',
                ),
                type='mmdet.PackDetInputs'),
        ],
        test_mode=True,
        type='TianzhibeiCarDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(metric='mAP', type='DOTAMetric')
val_pipeline = [
    dict(alpha_policy='warn', type='LoadGeoTiffRGB'),
    dict(box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
    dict(box_type_mapping=dict(gt_bboxes='rbox'), type='ConvertBoxType'),
    dict(size=1024, type='ResizeAndPad'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'geo_transform',
            'projection',
        ),
        type='mmdet.PackDetInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='RotLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
