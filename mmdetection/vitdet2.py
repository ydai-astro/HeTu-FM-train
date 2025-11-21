backbone_norm_cfg = dict(requires_grad=True, type='LN')
backend_args = None
batch_augments = [
    dict(pad_mask=True, size=(
        640,
        640,
    ), type='BatchFixedSizePad'),
]
#custom_hooks = [
#    dict(type='Fp16CompresssionHook'),
#]
custom_imports = dict(imports=[
    'projects.ViTDet.vitdet',
])
dataset_type = 'CocoDataset'
data_root = 'AI4Astronomy_v4/'
metainfo = {
    'classes': ('CJ', 'CS', 'FRI', 'FRII'),
    'palette': [
        (220, 20, 60),
        (119, 11, 32),
        (0, 0, 142),
        (106, 0, 228),
    ]
}

default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=1000,
        max_keep_ckpts=5,
        save_best='coco/bbox_mAP',
        save_last=True,
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))

launcher = 'pytorch'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        depth=12,
        drop_path_rate=0.1,
        embed_dim=768,
        img_size=640,
        pretrain_img_size=640,
        init_cfg=dict(
            checkpoint=
            '/mnt/workspace/mmselfsup/mae_type4.pth',
            type='Pretrained'),
        mlp_ratio=4,
        norm_cfg=dict(requires_grad=True, type='LN'),
        num_heads=12,
        patch_size=16,
        qkv_bias=True,
        type='ViT',
        use_rel_pos=True,
        window_block_indexes=[
            0,
            1,
            3,
            4,
            6,
            7,
            9,
            10,
        ],
        window_size=14),
    data_preprocessor=dict(
        batch_augments=[
            dict(pad_mask=True, size=(
                640,
                640,
            ), type='BatchFixedSizePad'),
        ],
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_mask=True,
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        backbone_channel=768,
        in_channels=[
            192,
            384,
            768,
            768,
        ],
        norm_cfg=dict(requires_grad=True, type='LN2d'),
        num_outs=5,
        out_channels=256,
        type='SimpleFPN'),
    roi_head=dict(
        bbox_head=dict(
            bbox_coder=dict(
                target_means=[
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                target_stds=[
                    0.1,
                    0.1,
                    0.2,
                    0.2,
                ],
                type='DeltaXYWHBBoxCoder'),
            conv_out_channels=256,
            fc_out_channels=1024,
            in_channels=256,
            loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
            loss_cls=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            norm_cfg=dict(requires_grad=True, type='LN2d'),
            num_classes=4,
            reg_class_agnostic=False,
            roi_feat_size=7,
            type='Shared4Conv1FCBBoxHead'),
        bbox_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=0, type='RoIAlign'),
            type='SingleRoIExtractor'),
        mask_head=dict(
            conv_out_channels=256,
            in_channels=256,
            loss_mask=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_mask=True),
            norm_cfg=dict(requires_grad=True, type='LN2d'),
            num_classes=4,
            num_convs=4,
            type='FCNMaskHead'),
        mask_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=14, sampling_ratio=0, type='RoIAlign'),
            type='SingleRoIExtractor'),
        type='StandardRoIHead'),
    rpn_head=dict(
        anchor_generator=dict(
            ratios=[
                0.5,
                1.0,
                2.0,
            ],
            scales=[
                8,
            ],
            strides=[
                4,
                8,
                16,
                32,
                64,
            ],
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[
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
            ],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
        loss_cls=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        num_convs=2,
        type='RPNHead'),
    test_cfg=dict(
        rcnn=dict(
            mask_thr_binary=0.5,
            max_per_img=100,
            nms=dict(iou_threshold=0.5, type='nms'),
            score_thr=0.01),
        rpn=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=1000)),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.5,
                neg_iou_thr=0.5,
                pos_iou_thr=0.5,
                type='MaxIoUAssigner'),
            debug=False,
            mask_size=28,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=True,
                neg_pos_ub=-1,
                num=512,
                pos_fraction=0.25,
                type='RandomSampler')),
        rpn=dict(
            allowed_border=-1,
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.3,
                neg_iou_thr=0.3,
                pos_iou_thr=0.7,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=False,
                neg_pos_ub=-1,
                num=256,
                pos_fraction=0.5,
                type='RandomSampler')),
        rpn_proposal=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=2000)),
    type='MaskRCNN')
norm_cfg = dict(requires_grad=True, type='LN2d')


##
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    #dict(type='Resize', scale=(680,680), keep_ratio=True,interpolation='bicubic'),
    dict(type='RandomResize', scale=(800, 800), ratio_range=(0.8,1), keep_ratio=True,interpolation='bicubic'),
    dict(type='Rotate', max_mag=10.0),####recompute_bbox=True
    dict(type='RandomCrop', crop_size=(640,640)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(640, 640), keep_ratio=True,interpolation='bicubic'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='PackDetInputs')
]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    #sampler=dict(type='ClassAwareSampler'),
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=train_pipeline,
        metainfo=metainfo,
        ann_file='annotations/Astronomy_instances_train.json',
        data_prefix=dict(img='')))
val_dataloader = dict(
    batch_size=4,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=test_pipeline,
        metainfo=metainfo,
        ann_file='annotations/Astronomy_instances_val.json',
        data_prefix=dict(img='')))
test_dataloader = val_dataloader
##

#
max_iters = 40000
warmup_iters = 500
optim_wrapper = dict(
    type='AmpOptimWrapper',
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg={
        'decay_rate': 0.7,
        'decay_type': 'layer_wise',
        'num_layers': 12,
    },
    optimizer=dict(
        type='AdamW',
        lr=0.001,
        betas=(0.9, 0.999),
        weight_decay=0.1,
    ))
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=warmup_iters),
    dict(
        type='CosineAnnealingLR',
        #T_max=max_iters-warmup_iters,
        by_epoch=False,
        begin=warmup_iters,
        end=max_iters)
]
#

resume = True


test_cfg = dict(type='TestLoop')
train_cfg = dict(max_iters=40000, type='IterBasedTrainLoop', val_interval=1000)
val_cfg = dict(type='ValLoop')

val_evaluator = dict(
    ann_file=data_root + 'annotations/Astronomy_instances_val.json',
    format_only=False,
    metric=[
        'bbox',
        'segm',
    ],
    type='CocoMetric')
test_evaluator = val_evaluator
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ])
work_dir = './work_dirs/vitdet2'
