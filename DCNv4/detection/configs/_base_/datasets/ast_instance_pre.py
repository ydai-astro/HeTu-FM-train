# dataset settings
dataset_type = 'CocoDataset'
data_root = '/mnt/workspace/mmdetection/AI4Astronomy_v4/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

classes=('CJ', 'CS', 'FRI', 'FRII')

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(480,480), keep_ratio=True,interpolation='bicubic'),
    dict(type='RandomCrop', crop_size=(448,448)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(448, 448),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True,interpolation='bicubic'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file='/mnt/workspace/DCNv4/detection/ast10w_label.json',
        img_prefix='/ast450w/',
        classes=classes,
        pipeline=train_pipeline),
    val=dict(

        type=dataset_type,
        ann_file=data_root + 'annotations/Astronomy_instances_val_v1.json',
        img_prefix=data_root,
        classes=classes,
        pipeline=test_pipeline),
    test=dict(

        type=dataset_type,
        ann_file=data_root + 'annotations/Astronomy_instances_val_v1.json',
        img_prefix=data_root,
        classes=classes,
        pipeline=test_pipeline))
    
evaluation = dict(metric=['bbox', 'segm'], classwise=True)
