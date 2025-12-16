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
git clone https://github.com/wenboyaoo/semseg-dinov3-pspnet
cd semseg-dinov3-pspnet
```
2. Install dependencies
```
pip install -r requirements.txt
```
3. Prepare dataset
Place the VOC dataset under:
```
dataset/vOC2012/
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

## Experiments and Ablation Studies

### Dataset

All experiments are conducted on **PASCAL VOC 2012** semantic segmentation benchmark (21 classes including background):

- Train: 10,582 images  
- Val: 1,449 images  

---

### Training Settings

- Input resolution: **224 × 224** (aligned with DINOv3 pretraining)
- Backbone: **DINOv3 ConvNeXt-Tiny (frozen)**
- Optimizer: SGD  
  - Initial LR: 0.01  
  - Momentum: 0.9  
  - Weight decay: 1e-4  
- LR schedule: Polynomial decay (power = 0.9)  
- Batch size: 32  
- Epochs: 50  
- Loss: Cross-entropy  
- Data augmentation:
  - Random scaling (0.5–2.0)
  - Random rotation (±10°)

---

### Evaluation Metrics

- mIoU
- Mean Accuracy (mAcc)
- Overall Accuracy (allAcc)

---

## Model Variants

We evaluate the following architectures to analyze the contribution of each component:

- **ResNet50 + PSPNet (baseline)**
- **DINOv3 + linear segmentation head**
- **DINOv3 + PPM**
- **DINOv3 + FPN**
- **DINOv3 + FPN + PPM**
- Different **FPN output channel dimensions**

The final architecture is:

DINOv3 ConvNeXt (frozen)
↓
FPN (OS = 8, configurable channels)
↓
PPM (1×1, 2×2, 3×3, 6×6)
↓
Segmentation Head

---

## Quantitative Results

### Ablation Results on VOC 2012 (Val)

| # | Model | mIoU | mAcc | allAcc |
|---|------|------|------|--------|
| 1 | ResNet50 + PPM (PSPNet) | 0.7705 | 0.8513 | 0.9489 |
| 2 | DINOv3 + Head | 0.7599 | 0.8634 | 0.9427 |
| 3 | DINOv3 + PPM | 0.7630 | 0.8654 | 0.9430 |
| 4 | DINOv3 + FPN (768 ch) | 0.7870 | 0.8738 | 0.9530 |
| 5 | DINOv3 + FPN + PPM (768 ch) | 0.8156 | 0.8989 | 0.9591 |
| 6 | ResNet50 + PPM (reproduced) | 0.7641 | 0.8483 | 0.9476 |
| 7 | ResNet50 (restored downsampling) + PPM | 0.7528 | 0.8361 | 0.9421 |
| 8 | DINOv3 + FPN + PPM (512 ch) | 0.8132 | 0.9028 | 0.9586 |
| 9 | DINOv3 + FPN + PPM (1024 ch) | 0.8162| 0.9061 | 0.9595 |
|10 | DINOv3 + FPN + PPM (2048 ch) | 0.8180 | 0.9062 | 0.9597 |

---

## Analysis

### Effect of PPM without Resolution Recovery

Directly attaching a PPM to frozen DINOv3 features brings only marginal improvement (Row 2 → 3).  
Compared with the original PSPNet, mIoU is lower while mAcc is higher, indicating degraded small-object discrimination.

We attribute this to the **large output stride (OS = 32)** of DINOv3 features, where spatial details are severely reduced and PPM pooling bins become excessively coarse.

---

### Importance of Output Stride (OS)

Restoring downsampling in the original PSPNet backbone (OS from 8 → 32) leads to a clear performance drop (Row 6 → 7), confirming that **high OS fundamentally limits PSPNet-style context aggregation**.

Since DINOv3 is designed to be used as a frozen backbone, structural modification or full fine-tuning is undesirable. Therefore, an external multi-scale fusion module is preferred.

---

### Effectiveness of FPN

Introducing an FPN to fuse multi-stage DINOv3 features and recover spatial resolution (OS = 8) significantly improves segmentation performance (Row 3 → 4 → 5).

Once the resolution bottleneck is removed, **PPM becomes effective again**, leading to consistent gains in mIoU and accuracy.

---

### Impact of FPN Channel Width

Increasing FPN output channels improves performance (512 → 768 → 1024), but gains saturate beyond 1024 channels.

Considering both accuracy and computational cost, **1024 channels** offer the best trade-off and are used as the default configuration.

---

## Conclusion

- **DINOv3 ConvNeXt** serves as a strong frozen feature extractor with robust semantic representations.
- **Output stride (OS)** is the key factor determining the effectiveness of PSPNet-style architectures.
- **FPN effectively resolves the resolution bottleneck** of DINOv3 and enables successful dense prediction.
- **PPM regains effectiveness only after multi-scale spatial information is restored.**
- The final model (**DINOv3 + FPN + PPM**, 1024 channels) achieves **mIoU 0.8162** on VOC 2012, outperforming the original PSPNet by approximately **+5.9% mIoU**.