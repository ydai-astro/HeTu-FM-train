# --------------------------------------------------------
# DCNv4
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/ast_instance.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py'
]
pretrained = 'https://huggingface.co/OpenGVLab/DCNv4/resolve/main/flash_intern_image_t_1k_224.pth'
#pretrained = '/mnt/workspace/mmselfsup/dcn4_type2.pth'
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
        init_cfg=dict(type='Pretrained', checkpoint=pretrained),
    ),
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
                num_classes=4,
                cls_predictor_cfg=dict(type='NormedLinear',tempearture=20),
                loss_cls=dict(type='SeesawLoss',p=0.8,q=2.0,num_classes=4,loss_weight=1.0)
                )
            ),
    test_cfg=dict(rcnn=dict(score_thr=0.0001,max_per_img=300))
)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2)



#lr_config = dict(_delete_=True,
#    policy='CosineAnnealing',
#    warmup='linear',
#    warmup_iters=500,
#    warmup_ratio=0.001,
#    min_lr_ratio=1e-8)



optimizer = dict(
    _delete_=True, type='AdamW', lr=0.0001, weight_decay=0.05,
    constructor='CustomLayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=30, layer_decay_rate=1.0,
                       depths=[4, 4, 18, 4]))
optimizer_config = dict(grad_clip=None)
# fp16 = dict(loss_scale=dict(init_scale=512))
evaluation = dict(save_best='auto',interval=4)
checkpoint_config = dict(
    interval=4,
    max_keep_ckpts=1,
    save_last=True,
)


#runner = dict(type='EpochBasedRunner', max_epochs=24)
