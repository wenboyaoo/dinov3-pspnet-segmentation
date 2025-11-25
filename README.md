# Semantic Segmentation with DINOv3 + PSPNet
Forked and modified from the original CVPR 2017 PSPNet implementation.
This version replaces the ResNet backbone with DINOv3 ConvNeXt-Tiny and integrates an FPN module to evaluate how frozen DINOv3 features perform inside the PSPNet framework

## Introduction

This repository is based on the original PSPNet codebase, with the following modifications:
- Replaced the ResNet backbone with facebook/dinov3-convnext-tiny-pretrain-lvd1689m (via HuggingFace Transformers)
- Added an FPN module to reduce output stride and increase channel capacity
- Updated the preprocessing pipeline to match the DINOv3 preprocessor

## Performance
- Dataset: PASCAL VOC 2012 Augmented
  - Train: 10,582 images
  - Val: 1,449 images
- Epochs: 50
- Other training configurations are listed in `config.yaml`

Validation result:
```
mIoU = 0.8162
mAcc = 0.8961
allAcc = 0.9595
```

# Reproduction
1. Clone
```
git clone https://github.com/xxx/semseg-dinov3-pspnet
cd semseg-dinov3-pspnet
```
2. Install dependencies
```
pip install -r requirements.txt
```
3. Prepare dataset
Place the VOC dataset under:
```
dataset/VOCdevkit/VOC2012/
```
4. Login to HuggingFace(required to access the gated DINOv3 models)
```
huggingface-cli login
```
5. Train
```
bash tool/train.sh
```
6. Evaluate
```
bash tool/eval.sh
```

## References
```
@misc{semseg2019,
  author={Zhao, Hengshuang},
  title={semseg},
  howpublished={\url{https://github.com/hszhao/semseg}},
  year={2019}
}

@inproceedings{zhao2017pspnet,
  title={Pyramid Scene Parsing Network},
  author={Zhao, Hengshuang and Shi, Jianping and Qi, Xiaojuan and Wang, Xiaogang and Jia, Jiaya},
  booktitle={CVPR},
  year={2017}
}

@inproceedings{zhao2018psanet,
  title={{PSANet}: Point-wise Spatial Attention Network for Scene Parsing},
  author={Zhao, Hengshuang and Zhang, Yi and Liu, Shu and Shi, Jianping and Loy, Chen Change and Lin, Dahua and Jia, Jiaya},
  booktitle={ECCV},
  year={2018}
}
```
