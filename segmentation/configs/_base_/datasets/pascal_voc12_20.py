# ------------------------------------------------------------------------------
# TCL
# Copyright (c) 2023 Kakao Brain. All Rights Reserved.
# ------------------------------------------------------------------------------
# Modified from GroupViT (https://github.com/NVlabs/GroupViT)
# Copyright (c) 2021-22, NVIDIA Corporation & affiliates. All Rights Reserved.
# ------------------------------------------------------------------------------
_base_ = ["../custom_import.py"]
# dataset settings
dataset_type = "PascalVOCDataset20"
data_root = "./data/VOCdevkit/VOC2012"
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(2048, 448),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
data = dict(
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="JPEGImages",
        ann_dir="SegmentationClass",
        split="ImageSets/Segmentation/val.txt",
        pipeline=test_pipeline,
    )
)

test_cfg = dict(mode="slide", stride=(224, 224), crop_size=(448, 448))
