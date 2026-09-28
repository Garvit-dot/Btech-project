default_scope = 'mmrotate'
dataset_type = 'DOTADataset'
data_root = (
    'C:/Users/LOQ/Documents/BtechProj/'
    'Btech-project/data/mmrotate_dota/'
)

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
    'working_length_rct',
)

metainfo = dict(
    classes=classes
)
file_client_args = dict(
    backend='disk'
)

train_pipeline = [
    dict(
        type='mmdet.LoadImageFromFile',
        file_client_args=file_client_args
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
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=[
            'horizontal',
            'vertical',
            'diagonal'
        ]
    ),

    dict(
        type='RandomRotate',
        prob=0.5,
        angle_range=180
    ),

    dict(
        type='mmdet.Pad',
        size=(640, 640),
        pad_val=dict(
            img=(114, 114, 114)
        )
    ),

    dict(
        type='mmdet.PackDetInputs'
    )
]

val_pipeline = [
    dict(
        type='mmdet.LoadImageFromFile',
        file_client_args=file_client_args
    ),

    dict(
        type='mmdet.Resize',
        scale=(640, 640),
        keep_ratio=True
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
        type='mmdet.Pad',
        size=(640, 640),
        pad_val=dict(
            img=(114, 114, 114)
        )
    ),

    dict(
        type='mmdet.PackDetInputs',
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor'
        )
    )
]

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

checkpoint = (
    'https://download.openmmlab.com/mmdetection/v3.0/'
    'rtmdet/cspnext_rsb_pretrain/'
    'cspnext-tiny_imagenet_600e.pth'
)

angle_version = 'le90'

model = dict(

    type='mmdet.RTMDet',

    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        boxtype2tensor=False,
        batch_augments=None
    ),

    backbone=dict(
        type='mmdet.CSPNeXt',
        arch='P5',
        expand_ratio=0.5,

        deepen_factor=0.167,
        widen_factor=0.375,

        channel_attention=True,

        norm_cfg=dict(
            type='SyncBN'
        ),

        act_cfg=dict(
            type='SiLU'
        ),

        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint=checkpoint
        )
    ),

    neck=dict(
        type='mmdet.CSPNeXtPAFPN',

        in_channels=[
            96,
            192,
            384
        ],

        out_channels=96,

        num_csp_blocks=1,

        expand_ratio=0.5,

        norm_cfg=dict(
            type='SyncBN'
        ),

        act_cfg=dict(
            type='SiLU'
        )
    ),

    bbox_head=dict(

        type='RotatedRTMDetSepBNHead',

        num_classes=14,

        in_channels=96,

        stacked_convs=2,

        feat_channels=96,

        angle_version=angle_version,

        anchor_generator=dict(
            type='mmdet.MlvlPointGenerator',
            offset=0,
            strides=[
                8,
                16,
                32
            ]
        ),

        bbox_coder=dict(
            type='DistanceAnglePointCoder',
            angle_version=angle_version
        ),

        loss_cls=dict(
            type='mmdet.QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0
        ),

        loss_bbox=dict(
            type='RotatedIoULoss',
            mode='linear',
            loss_weight=2.0
        ),

        with_objectness=False,

        exp_on_reg=False,

        share_conv=True,

        pred_kernel_size=1,

        use_hbbox_loss=False,

        scale_angle=False,

        loss_angle=None,

        norm_cfg=dict(
            type='SyncBN'
        ),

        act_cfg=dict(
            type='SiLU'
        )
    ),

    train_cfg=dict(

        assigner=dict(
            type='mmdet.DynamicSoftLabelAssigner',

            iou_calculator=dict(
                type='RBboxOverlaps2D'
            ),

            topk=13
        ),

        allowed_border=-1,

        pos_weight=-1,

        debug=False
    ),

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

# ------------------------------------------------------------
# Training dataloader
# ------------------------------------------------------------

train_dataloader = dict(

    batch_size=2,

    num_workers=0,

    persistent_workers=False,

    sampler=dict(
        type='DefaultSampler',
        shuffle=True
    ),

    batch_sampler=None,

    pin_memory=False,

    dataset=dict(

        type=dataset_type,

        data_root=data_root,

        ann_file='train/annfiles/',

        data_prefix=dict(
            img_path='train/images/'
        ),

        img_shape=(1100, 795),

        filter_cfg=dict(
            filter_empty_gt=True
        ),

        metainfo=metainfo,

        pipeline=train_pipeline
    )
)

# ------------------------------------------------------------
# Validation dataloader
# ------------------------------------------------------------

val_dataloader = dict(

    batch_size=1,

    num_workers=0,

    persistent_workers=False,

    drop_last=False,

    sampler=dict(
        type='DefaultSampler',
        shuffle=False
    ),

    dataset=dict(

        type=dataset_type,

        data_root=data_root,

        ann_file='val/annfiles/',

        data_prefix=dict(
            img_path='val/images/'
        ),

        img_shape=(1100, 795),

        test_mode=True,

        metainfo=metainfo,

        pipeline=val_pipeline
    )
)

test_dataloader = val_dataloader

# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

val_evaluator = dict(
    type='DOTAMetric',
    metric='mAP',
    iou_thrs=0.5,
    eval_mode='11points'
)

test_evaluator = val_evaluator

# ------------------------------------------------------------
# Training schedule
# ------------------------------------------------------------

train_cfg = dict(
    type='EpochBasedTrainLoop',

    max_epochs=50,

    val_interval=5
)

val_cfg = dict(
    type='ValLoop'
)

test_cfg = dict(
    type='TestLoop'
)

# ------------------------------------------------------------
# Learning rate scheduler
# ------------------------------------------------------------

base_lr = 0.0000625

param_scheduler = [

    dict(
        type='LinearLR',

        start_factor=1.0e-5,

        by_epoch=False,

        begin=0,

        end=500
    ),

    dict(
        type='CosineAnnealingLR',

        eta_min=base_lr * 0.05,

        begin=25,

        end=50,

        T_max=25,

        by_epoch=True,

        convert_to_iter_based=True
    )
]

# ------------------------------------------------------------
# Optimizer
# ------------------------------------------------------------

optim_wrapper = dict(

    type='OptimWrapper',

    optimizer=dict(

        type='AdamW',

        lr=base_lr,

        weight_decay=0.05
    ),

    paramwise_cfg=dict(

        norm_decay_mult=0,

        bias_decay_mult=0,

        bypass_duplicate=True
    )
)

# ------------------------------------------------------------
# Runtime
# ------------------------------------------------------------

work_dir = (
    './runs/mmrotate/rtmdet_tiny_r_iopa'
)

env_cfg = dict(

    cudnn_benchmark=False,

    mp_cfg=dict(
        mp_start_method='spawn',
        opencv_num_threads=0
    ),

    dist_cfg=dict(
        backend='gloo'
    )
)

default_hooks = dict(

    timer=dict(
        type='IterTimerHook'
    ),

    logger=dict(
        type='LoggerHook',
        interval=20
    ),

    param_scheduler=dict(
        type='ParamSchedulerHook'
    ),

    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        max_keep_ckpts=3
    ),

    sampler_seed=dict(
        type='DistSamplerSeedHook'
    ),

    visualization=dict(
        type='mmdet.DetVisualizationHook'
    )
)

log_level = 'INFO'

# Keep pretrained backbone initialization.
load_from = None

resume = False	