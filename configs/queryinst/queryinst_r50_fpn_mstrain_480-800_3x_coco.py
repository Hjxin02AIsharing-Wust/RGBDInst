_base_ = './queryinst_r50_fpn_1x_coco.py'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
min_values = (480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        img_scale=[(1333, value) for value in min_values],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]

lr_config = dict(policy='step', step=[27, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)

dataset_type = 'CocoDataset'
classes = ('surface',)
data_root = '/media/rcus-cv/data/hjx/datasets/DTSCD/40000train_4000val/'
data = dict(
    # _delete_=True,
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'train/train20000_real.json',
        img_prefix=data_root + 'train/coco_data/',
        pipeline=train_pipeline,),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'val/val2000_real.json',
        img_prefix=data_root + 'val/coco_data/'),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'val/val2000_real.json',
        img_prefix=data_root + 'val/coco_data/'))