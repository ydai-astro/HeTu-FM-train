# dataset settings
dataset_type = 'CocoDataset'
data_root = '/ast450w/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

classes=('CJ', 'CS', 'FRI', 'FRII')

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
    samples_per_gpu=1024,
    workers_per_gpu=8,
    val=dict(
        type=dataset_type,
        ann_file='/mnt/workspace/DCNv4/detection/ast450w.json',
        img_prefix=data_root,
        classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='/mnt/workspace/DCNv4/detection/ast450w.json',
        img_prefix=data_root,
        classes=classes,
        pipeline=test_pipeline))
    
evaluation = dict(metric=['bbox', 'segm'], classwise=True)
