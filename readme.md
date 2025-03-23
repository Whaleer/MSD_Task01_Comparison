# MSD Task01 Segmentation Models Comparison

## Overview
This repository contains implementations of various state-of-the-art deep learning models for 3D medical image segmentation, specifically focused on the Medical Segmentation Decathlon (MSD) Task01 (Brain Tumor Segmentation). The code provides a comprehensive framework for comparing different segmentation approaches on the same dataset.

## Implemented Models

### 1. Attention U-Net
An extension of the standard U-Net architecture that incorporates attention gates to focus on relevant target structures of varying shapes and sizes.

### 2. SegResNet
A powerful segmentation network that combines residual blocks with a U-Net-like encoder-decoder structure, offering improved gradient flow and feature representation.

### 3. 3D U-Net
The 3D adaptation of the classic U-Net architecture, specifically designed for volumetric medical image segmentation.

### 4. UNETR (UNEt TRansformer)
A novel architecture that leverages transformer-based encoders combined with CNN decoders, effectively capturing global context while maintaining spatial information.

### 5. SwinUNETR
An advanced model that integrates the Swin Transformer into a U-Net-like structure, providing hierarchical feature representation with shifted windows for efficient and effective segmentation.

## Usage
Each model implementation is contained in its respective directory with training and inference scripts. Please refer to the specific model directories for detailed usage instructions.

## Requirements
- Python 3.7+
- PyTorch 1.9+
- MONAI 0.8+
- Numpy, Scipy, Matplotlib, etc.

## Acknowledgements
All implementations are based on the MONAI framework (Medical Open Network for AI), an open-source foundation for deep learning in healthcare imaging.

## Citation
If you find this code useful for your research, please consider citing the original MONAI framework:

```bibtex
@article{cardoso2022monai,
  title={Monai: An open-source framework for deep learning in healthcare},
  author={Cardoso, M Jorge and Li, Wenqi and Brown, Richard and Ma, Nic and Kerfoot, Eric and Wang, Yiheng and Murrey, Benjamin and Myronenko, Andriy and Zhao, Can and Yang, Dong and others},
  journal={arXiv preprint arXiv:2211.02701},
  year={2022}
}
