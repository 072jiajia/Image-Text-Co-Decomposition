# Image-Text Co-Decomposition
Code for CVPR 2024 Image-Text Co-Decomposition for Text-Supervised Semantic Segmentation

[Paper](https://arxiv.org/abs/2309.03900)  

## Notes on environment setup

1. It is best to start a new conda environment.
```
conda create -n [env_name] python=3.10
conda activate [env_name]
```

2. Install required packages
```shell
pip install torch==1.12.1 torchvision==0.13.1
pip install -U openmim
pip install -r requirements.txt
python -m mim install mmcv-full==1.6.2 mmsegmentation==0.27.0
```

3. Deal with NLTK stuffs
```python
python
>>> import nltk
>>> nltk.download('punkt')
>>> nltk.download('averaged_perceptron_tagger')
```

## Datasets

Note that much of this section is adapted from the [data preparation section of TCL README](https://github.com/kakaobrain/tcl#datasets).

### Training datasets

In training, we use Conceptual Caption 3m and 12m. We use [img2dataset](https://github.com/rom1504/img2dataset) tool and follow [these instructions](https://github.com/kakaobrain/tcl#training-datasets) to download and preprocess the datasets.

### Evaluation datasets

In the paper, we use 6 benchmarks; PASCAL VOC, PASCAL Context, and COCO-Object, COCO-Stuff, Cityscapes, and ADE20k. We need to prepare 5 datasets: PASCAL VOC, PASCAL Context, COCO-Stuff164k, Cityscapes, and ADE20k.

Please download and setup [PASCAL VOC](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#pascal-voc), [PASCAL Context](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#pascal-context), [COCO-Stuff164k](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#coco-stuff-164k), [Cityscapes](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#cityscapes), and [ADE20k](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#ade20k) datasets following [MMSegmentation data preparation document](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md).

#### COCO Object

COCO-Object dataset uses only object classes from COCO-Stuff164k dataset by collecting instance semgentation annotations.
Run the following command to convert instance segmentation annotations to semantic segmentation annotations:

```shell
python convert_dataset/convert_coco.py data/coco_stuff164k/ -o data/coco_stuff164k/
```

The overall file structure is as follows:

```shell
ImageTextCoDecomposition
├── data
│   ├── gcc3m
│   │   ├── gcc-train-000000.tar
│   │   ├── ...
│   ├── gcc12m
│   │   ├── cc-000000.tar
│   │   ├── ...
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   ├── VOCdevkit
│   │   ├── VOC2012
│   │   │   ├── JPEGImages
│   │   │   ├── SegmentationClass
│   │   │   ├── ImageSets
│   │   │   │   ├── Segmentation
│   │   ├── VOC2010
│   │   │   ├── JPEGImages
│   │   │   ├── SegmentationClassContext
│   │   │   ├── ImageSets
│   │   │   │   ├── SegmentationContext
│   │   │   │   │   ├── train.txt
│   │   │   │   │   ├── val.txt
│   │   │   ├── trainval_merged.json
│   │   ├── VOCaug
│   │   │   ├── dataset
│   │   │   │   ├── cls
│   ├── ade
│   │   ├── ADEChallengeData2016
│   │   │   ├── annotations
│   │   │   │   ├── training
│   │   │   │   ├── validation
│   │   │   ├── images
│   │   │   │   ├── training
│   │   │   │   ├── validation
│   ├── coco_stuff164k
│   │   ├── images
│   │   │   ├── train2017
│   │   │   ├── val2017
│   │   ├── annotations
│   │   │   ├── train2017
│   │   │   ├── val2017
```

## Run Experiments
Follow the original installation process to setup your environment, then run the following script
```
  sh scripts/train.sh
```

## Evaluation

We provide our [official checkpoint](https://drive.google.com/file/d/1CpaaZswztgVulpTdVl_eiBRaIeoVAGJ2/view?usp=sharing) to reproduce the main results of our paper.

- Evaluation

```
  sh scripts/eval.sh
```

## Citation

```bibtex
@article{wu2024image,
  title={Image-Text Co-Decomposition for Text-Supervised Semantic Segmentation},
  author={Wu, Ji-Jia and Chang, Andy Chia-Hao and Chuang, Chieh-Yu and Chen, Chun-Pei and Liu, Yu-Lun and Chen, Min-Hung and Hu, Hou-Ning and Chuang, Yung-Yu and Lin, Yen-Yu},
  journal={arXiv preprint arXiv:2404.04231},
  year={2024}
}
```
