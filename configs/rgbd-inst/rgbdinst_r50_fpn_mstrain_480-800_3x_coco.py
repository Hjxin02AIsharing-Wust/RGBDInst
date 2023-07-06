_base_ = './queryinst_r50_fpn_1x_coco.py'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53, 2], std=[58.395, 57.12, 57.375, 1], to_rgb=True)
min_values = (480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800)
train_pipeline = [
    dict(type='LoadImageandDepthFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        img_scale=[(512, value) for value in min_values],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]

test_pipeline = [
    dict(type='LoadImageandDepthFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(1333, 800),
        img_scale=(640, 480),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

lr_config = dict(policy='step', step=[27, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)

dataset_type = 'CocoDataset'
classes = ('surface',)
data_root = 'D:/AIRS-dataset/4000train_400val/'
data = dict(
    # _delete_=True,
    samples_per_gpu=3,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'train/train_4000.json',
        img_prefix=data_root + 'train/coco_data/',
        pipeline=train_pipeline,),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'val/val_400.json',
        img_prefix=data_root + 'val/coco_data/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'val/val_400.json',
        img_prefix=data_root + 'val/coco_data/',
        pipeline=test_pipeline))
