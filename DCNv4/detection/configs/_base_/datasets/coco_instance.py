# dataset settings
dataset_type = 'CocoDataset'
data_root = '/mnt/workspace/mmdetection/AI4Astronomy_v4/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(480,480), keep_ratio=True,interpolation='bicubic'),
    #dict(type='RandomResize', scale=(560, 560), ratio_range=(0.8,1), keep_ratio=True,interpolation='bicubic'),
    #dict(type='Rotate', max_mag=5.0),####recompute_bbox=True
    dict(type='RandomCrop', crop_size=(448,448)),
    dict(type='RandomFlip', prob=0.5),
    #dict(type='Normalize', **img_norm_cfg),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(448, 448), keep_ratio=True,interpolation='bicubic'),
    #dict(type='Normalize',  **img_norm_cfg),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='PackDetInputs')
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox', 'segm'], classwise=True)
