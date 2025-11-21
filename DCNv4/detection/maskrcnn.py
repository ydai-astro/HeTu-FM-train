# --------------------------------------------------------
# DCNv4
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
_base_ = [
    'configs/_base_/models/mask_rcnn_r50_fpn.py',
    #'../_base_/datasets/coco_instance.py',
    #'../_base_/schedules/schedule_3x.py',
    'configs/_base_/default_runtime.py'
]
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
pretrained = 'https://huggingface.co/OpenGVLab/DCNv4/resolve/main/flash_intern_image_t_1k_224.pth'
model = dict(
    backbone=dict(
        _delete_=True,
        type='FlashInternImage',
        core_op='DCNv4',
        channels=64,
        depths=[4, 4, 18, 4],
        groups=[4, 8, 16, 32],
        mlp_ratio=4.,
        drop_path_rate=0.2,
        norm_layer='LN',
        layer_scale=1.0,
        offset_scale=1.0,
        post_norm=False,
        with_cp=True,
        out_indices=(0, 1, 2, 3),
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    # We leverage the FPN implemented in ViTDet for stable training,
    # and we don't benefit from this FPN in terms of performance.
    neck=dict(
        type='FPN_vitdet',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        norm_cfg=dict(type='LN', requires_grad=True),
        use_residual=True,
        num_outs=5),
    roi_head=dict(
            mask_head=dict(num_classes=4),
            bbox_head=dict(
                num_classes=4
                )
            )
    )
# By default, models are trained on 8 GPUs with 2 images per GPU
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(512,512), keep_ratio=True,interpolation='bicubic'),
    dict(type='RandomCrop', crop_size=(480,480)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(480, 480), keep_ratio=True,interpolation='bicubic'),
    dict(type='Normalize',  **img_norm_cfg),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='PackDetInputs')
]
# we use 4 nodes to train this model, with a total batch size of 64
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


###############
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=4)
val_cfg = dict(type='ValLoop')


optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.05,constructor='CustomLayerDecayOptimizerConstructor',paramwise_cfg=dict(num_layers=30, layer_decay_rate=1.0,depths=[4, 4, 18, 4]))
optimizer_config = dict(grad_clip=None)
# fp16 = dict(loss_scale=dict(init_scale=512))
evaluation = dict(save_best='auto')
checkpoint_config = dict(
    interval=4,
    max_keep_ckpts=1,
    save_last=True,
)



# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24, val_interval=4)