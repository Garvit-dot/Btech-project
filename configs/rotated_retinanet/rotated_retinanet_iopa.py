_base_ = []

# ============================================================
# DATASET
# ============================================================

dataset_type = 'DOTADataset'

data_root = 'data/mmrotate_dota/'

classes = (
    'GP',
    'boneloss',
    'caries',
    'crown',
    'dental_stone',
    'erupting_tooth',
    'filled',
    'healing_socket',
    'missing',
    'opturation',
    'periapical_lesion',
    'pulp_stone',
    'rootcanal',
    'working_length_rct'
)

metainfo = dict(
    classes=classes
)

angle_version = 'le90'


# ============================================================
# MODEL
# ============================================================

model = dict(
    type='mmdet.RetinaNet',

    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        boxtype2tensor=False
    ),

    # --------------------------------------------------------
    # BACKBONE
    # --------------------------------------------------------

    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,

        norm_cfg=dict(
            type='BN',
            requires_grad=True
        ),

        norm_eval=True,
        style='pytorch',

        init_cfg=dict(
            type='Pretrained',
            checkpoint='torchvision://resnet50'
        )
    ),

    # --------------------------------------------------------
    # FPN
    # --------------------------------------------------------

    neck=dict(
        type='mmdet.FPN',

        in_channels=[
            256,
            512,
            1024,
            2048
        ],

        out_channels=256,

        start_level=1,

        add_extra_convs='on_input',

        num_outs=5
    ),

    # --------------------------------------------------------
    # ROTATED RETINANET HEAD
    # --------------------------------------------------------

    bbox_head=dict(
        type='mmdet.RetinaHead',

        num_classes=14,

        in_channels=256,

        stacked_convs=4,

        feat_channels=256,

        # ----------------------------------------------------
        # Rotated Anchor Generator
        # ----------------------------------------------------

        anchor_generator=dict(
            type='FakeRotatedAnchorGenerator',

            angle_version=angle_version,

            octave_base_scale=4,

            scales_per_octave=3,

            ratios=[
                1.0,
                0.5,
                2.0
            ],

            strides=[
                8,
                16,
                32,
                64,
                128
            ]
        ),

        # ----------------------------------------------------
        # Rotated Bounding Box Coder
        # ----------------------------------------------------

        bbox_coder=dict(
            type='DeltaXYWHTRBBoxCoder',

            target_means=(
                0,
                0,
                0,
                0,
                0
            ),

            target_stds=(
                1,
                1,
                1,
                1,
                1
            ),

            angle_version=angle_version,

            norm_factor=None,

            edge_swap=True,

            proj_xy=True
        ),

        # ----------------------------------------------------
        # Classification Loss
        # ----------------------------------------------------

        loss_cls=dict(
            type='mmdet.FocalLoss',

            use_sigmoid=True,

            gamma=2.0,

            alpha=0.25,

            loss_weight=1.0
        ),

        # ----------------------------------------------------
        # Bounding Box Regression Loss
        # ----------------------------------------------------

        loss_bbox=dict(
            type='mmdet.L1Loss',

            loss_weight=1.0
        )
    ),

    # ========================================================
    # TRAINING CONFIG
    # ========================================================

    train_cfg=dict(

        assigner=dict(
            type='mmdet.MaxIoUAssigner',

            pos_iou_thr=0.5,

            neg_iou_thr=0.4,

            min_pos_iou=0,

            ignore_iof_thr=-1,

            iou_calculator=dict(
                type='RBboxOverlaps2D'
            )
        ),

        sampler=dict(
            type='mmdet.PseudoSampler'
        ),

        allowed_border=-1,

        pos_weight=-1,

        debug=False
    ),

    # ========================================================
    # MODEL TEST CONFIG
    # ========================================================

    test_cfg=dict(

        nms_pre=2000,

        min_bbox_size=0,

        score_thr=0.05,

        nms=dict(
            type='nms_rotated',

            iou_threshold=0.1
        ),

        max_per_img=2000
    )
)


# ============================================================
# TRAINING PIPELINE
# ============================================================

train_pipeline = [

    dict(
        type='mmdet.LoadImageFromFile'
    ),

    dict(
        type='mmdet.LoadAnnotations',

        with_bbox=True,

        box_type='qbox'
    ),

    # Convert quadrilateral boxes to rotated boxes
    dict(
        type='ConvertBoxType',

        box_type_mapping=dict(
            gt_bboxes='rbox'
        )
    ),

    dict(
        type='mmdet.Resize',

        scale=(640, 640),

        keep_ratio=True
    ),

    dict(
        type='mmdet.PackDetInputs',

        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale',
            'scale_factor',
            'keep_ratio',
            'homography_matrix'
        )
    )
]


# ============================================================
# VALIDATION PIPELINE
# ============================================================

val_pipeline = [

    dict(
        type='mmdet.LoadImageFromFile'
    ),

    dict(
        type='mmdet.LoadAnnotations',

        with_bbox=True,

        box_type='qbox'
    ),

    dict(
        type='ConvertBoxType',

        box_type_mapping=dict(
            gt_bboxes='rbox'
        )
    ),

    dict(
        type='mmdet.Resize',

        scale=(640, 640),

        keep_ratio=True
    ),

    dict(
        type='mmdet.PackDetInputs',

        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale',
            'scale_factor',
            'keep_ratio',
            'homography_matrix'
        )
    )
]


# ============================================================
# TRAIN DATALOADER
# ============================================================

train_dataloader = dict(

    batch_size=2,

    num_workers=0,

    persistent_workers=False,

    sampler=dict(
        type='DefaultSampler',

        shuffle=True
    ),

    dataset=dict(

        type='DOTADataset',

        data_root=data_root,

        ann_file='train/annfiles/',

        data_prefix=dict(
            img_path='train/images/'
        ),

        metainfo=metainfo,

        pipeline=train_pipeline,

        filter_cfg=dict(
            filter_empty_gt=False
        )
    )
)


# ============================================================
# VALIDATION DATALOADER
# ============================================================

val_dataloader = dict(

    batch_size=1,

    num_workers=0,

    persistent_workers=False,

    sampler=dict(
        type='DefaultSampler',

        shuffle=False
    ),

    dataset=dict(

        type='DOTADataset',

        data_root=data_root,

        ann_file='val/annfiles/',

        data_prefix=dict(
            img_path='val/images/'
        ),

        metainfo=metainfo,

        pipeline=val_pipeline,

        test_mode=True
    )
)


test_dataloader = val_dataloader


# ============================================================
# EVALUATION
# ============================================================

val_evaluator = dict(

    type='DOTAMetric',

    iou_thrs=[0.5],

    eval_mode='11points'
)

test_evaluator = val_evaluator


# ============================================================
# TRAINING LOOP
# ============================================================

train_cfg = dict(

    type='EpochBasedTrainLoop',

    # FINAL TRAINING
    max_epochs=50,

    # Validate every 5 epochs
    val_interval=5
)


val_cfg = dict(
    type='ValLoop'
)

test_cfg = dict(
    type='TestLoop'
)


# ============================================================
# OPTIMIZER
# ============================================================

optim_wrapper = dict(

    type='OptimWrapper',

    optimizer=dict(

        type='AdamW',

        lr=1e-4,

        weight_decay=1e-4
    ),

    clip_grad=dict(

        max_norm=35,

        norm_type=2
    )
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

param_scheduler = [

    # Warm-up
    dict(

        type='LinearLR',

        start_factor=0.001,

        by_epoch=False,

        begin=0,

        end=500
    ),

    # Learning-rate decay
    dict(

        type='MultiStepLR',

        begin=0,

        end=50,

        by_epoch=True,

        milestones=[
            35,
            45
        ],

        gamma=0.1
    )
]


# ============================================================
# MMROTATE DEFAULT SCOPE
# ============================================================

default_scope = 'mmrotate'


# ============================================================
# DEFAULT HOOKS
# ============================================================
#
# IMPORTANT:
#
# Do NOT add a visualization hook here.
#
# The installed MMDetection/MMEngine combination previously
# caused:
#
# Visualizer.add_datasample()
# unexpected keyword argument 'pred_score_thr'
#
# Therefore visualization is intentionally omitted.
#
# ============================================================

default_hooks = dict(

    runtime_info=dict(
        type='RuntimeInfoHook'
    ),

    timer=dict(
        type='IterTimerHook'
    ),

    sampler_seed=dict(
        type='DistSamplerSeedHook'
    ),

    logger=dict(
        type='LoggerHook',

        interval=50
    ),

    param_scheduler=dict(
        type='ParamSchedulerHook'
    ),

    checkpoint=dict(

        type='CheckpointHook',

        # Save every 5 epochs
        interval=5,

        save_best='auto',

        max_keep_ckpts=3
    )
)


# ============================================================
# ENVIRONMENT
# ============================================================

env_cfg = dict(

    cudnn_benchmark=False,

    mp_cfg=dict(

        mp_start_method='spawn',

        opencv_num_threads=0
    ),

    dist_cfg=dict(

        backend='nccl'
    )
)


# ============================================================
# LOGGING
# ============================================================

log_processor = dict(

    type='LogProcessor',

    window_size=50,

    by_epoch=True
)

log_level = 'INFO'


# ============================================================
# CHECKPOINT / RESUME
# ============================================================

load_from = None

resume = False


# ============================================================
# WORK DIRECTORY
# ============================================================

work_dir = './runs/mmrotate/rotated_retinanet_iopa'