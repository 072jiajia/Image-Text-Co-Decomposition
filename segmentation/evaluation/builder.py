# ------------------------------------------------------------------------------
# CoDe
# Copyright (C) 2024 by Ji-Jia Wu. All Rights Reserved.
# ------------------------------------------------------------------------------
# Modified from TCL (https://github.com/kakaobrain/tcl)
# Copyright (c) 2023 Kakao Brain. All Rights Reserved.
# ------------------------------------------------------------------------------
import mmcv
import torch
from mmseg.datasets import build_dataloader, build_dataset
from datasets import get_template

from .tcl_seg import TCLSegInference


def build_dataset_class_tokens(text_transform, template_set, classnames):
    tokens = []
    templates = get_template(template_set)
    for classname in classnames:
        tokens.append(
            torch.stack([text_transform(template.format(classname)) for template in templates])
        )
    # [N, T, L], N: number of instance, T: number of captions (including ensembled), L: sequence length
    tokens = torch.stack(tokens)

    return tokens


def build_dataset_class_sentences(template_set, classnames):
    sentences = []
    templates = get_template(template_set)
    for classname in classnames:
        sentences.append([template.format(classname)
                         for template in templates])
    return sentences


def build_seg_dataset(config):
    """Build a dataset from config."""
    cfg = mmcv.Config.fromfile(config)
    dataset = build_dataset(cfg.data.test)
    return dataset


def build_seg_dataloader(dataset):
    # batch size is set to 1 to handle varying image size (due to different aspect ratio)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=1,
        dist=True,
        shuffle=False,
        persistent_workers=True,
        pin_memory=False,
    )
    return data_loader


def build_seg_inference(
    model,
    dataset,
    config,
    seg_config,
):
    dset_cfg = mmcv.Config.fromfile(seg_config)  # dataset config
    with_bg = dataset.CLASSES[0] == "background"
    if with_bg:
        classnames = dataset.CLASSES[1:]
    else:
        classnames = dataset.CLASSES

    text_embedding = model.build_text_embedding_from_noun(classnames)
    text_tokens = build_dataset_class_sentences(
        config.evaluate.template, classnames)
    kp_branch_text_embedding = model.build_text_embedding_from_text_tokens(text_tokens)

    kwargs = dict(with_bg=with_bg)
    if hasattr(dset_cfg, "test_cfg"):
        kwargs["test_cfg"] = dset_cfg.test_cfg

    seg_model = TCLSegInference(
        model, text_embedding, kp_branch_text_embedding, **kwargs, **config.evaluate)

    seg_model.CLASSES = dataset.CLASSES
    seg_model.PALETTE = dataset.PALETTE

    return seg_model
