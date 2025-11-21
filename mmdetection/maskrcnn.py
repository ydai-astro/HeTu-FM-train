_base_ = ['configs/_base_/models/mask-rcnn_r50_fpn.py']
test_cfg = dict(type='TestLoop')

custom_imports = dict(
    imports=['mmpretrain.models'], allow_failed_imports=False)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-base_3rdparty-fcmae_in1k_20230104-8a798eaf.pth'  # noqa

model = dict(
    backbone=dict(
        _delete_=True,
        type='mmpretrain.ConvNeXt',
        arch='base',
        out_indices=[0, 1, 2, 3],
        # TODO: verify stochastic depth rate {0.1, 0.2, 0.3, 0.4}
        drop_path_rate=0.4,
        layer_scale_init_value=0.,  # disable layer scale when using GRN
        gap_before_final_norm=False,
        use_grn=True,  # V2 uses GRN
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    neck=dict(in_channels=[128, 256, 512, 1024]),
    roi_head=dict(
          mask_head=dict(num_classes=4),
          bbox_head=dict(
          num_classes=4,
          
               )
            ),
    test_cfg=dict(
        rpn=dict(nms=dict(type='nms')),  # TODO: does RPN use soft_nms?
        rcnn=dict(nms=dict(type='soft_nms'))))

#data

    
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

backend_args = None

#preprocess
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(480,480), keep_ratio=True,interpolation='bicubic'),
    #dict(type='RandomResize', scale=(480, 480), ratio_range=(0.94,1), keep_ratio=True,interpolation='bicubic'),
    #dict(type='Rotate', max_mag=10.0),
    dict(type='RandomCrop', crop_size=(448,448)),
    dict(type='RandomFlip', direction=['horizontal'],prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(448, 448), keep_ratio=True,interpolation='bicubic'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='PackDetInputs')
]
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    #sampler=dict(type='ClassAwareSampler'),
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=train_pipeline,
        metainfo=metainfo,
        ann_file='annotations/Astronomy_instances_train.json',
        data_prefix=dict(img='')))
val_dataloader = dict(
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=test_pipeline,
        metainfo=metainfo,
        ann_file='annotations/Astronomy_instances_val.json',
        data_prefix=dict(img='')))
test_dataloader = val_dataloader
#eval
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/Astronomy_instances_val.json',
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args=backend_args)

test_evaluator = val_evaluator

#LR
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=4)
val_cfg = dict(type='ValLoop')


param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=True, begin=0, end=2,convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=22,
        by_epoch=True,
        begin=2,
        end=24,
        convert_to_iter_based=True)
]


#optimizer=dict(type='AdamW',lr=0.0001,betas=(0.9, 0.999),weight_decay=0.05,))
#optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0001))
auto_scale_lr = dict(enable=False, base_batch_size=8)
randomness = dict(seed=0, diff_rank_seed=True)

default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=4),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=100, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False


visualization=dict( #用户可视化验证和测试结果
    type='DetVisualizationHook',
    draw=True,
    interval=1,
    show=False)