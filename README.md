# Improved Scene Classification by Dynamic CNNs

This repository contains the implementation of the paper **“Improved Scene Classification by Dynamic CNNs.”**

The proposed approach builds upon GLNet as the baseline framework and improves remote sensing scene classification under clear and cloudy conditions through dynamic convolution.

## Overview

The method is based on the GLNet architecture proposed for remote sensing scene classification under clear and cloudy environments. Dynamic convolution modules are incorporated into the network to improve its adaptability to varying scene and cloud conditions.

## Requirements

- Python 3.6+
- PyTorch 1.0+
- torchvision
- NumPy
- Pillow

The required packages can be installed using:

```bash
pip install torch torchvision numpy pillow
```

## Dataset

The dataset used in this study can be downloaded from the link provided by our baseline GLNet repository:

[Download the dataset from Google Drive](https://drive.google.com/file/d/1F_68mh40vNLOwila32GBYNHVEZI1HiTT/view)

After downloading, place the dataset under the following directory:

```text
data/
```

## Pretrained Models

The trained DY-GLNet and DY-ResNet50 models can be downloaded from:

[Download DY-GLNet and DY-ResNet50 models from Google Drive](https://drive.google.com/drive/folders/1JUNKdFGRo5YeT6jGnEDdyIZ-V1u-Wutb?usp=sharing)

After downloading the models, update the corresponding checkpoint paths in the Python scripts.

## Usage

Clone the repository:

```bash
git clone https://github.com/EEAkbaba/DY-GLNet.git
cd DY-GLNet
```

### DY-ResNet50 Training

To train the DY-ResNet50 model:

```bash
python baseline.py
```

### DY-GLNet Training

To train the proposed DY-GLNet model:

```bash
python train.py
```

Before training, configure the dataset paths and training settings in the corresponding scripts.

## Repository Structure

```text
DY-GLNet/
├── baseline.py
├── cloud_generation.py
├── feature_extractor.py
├── model.py
├── RS_Dataset.py
├── train.py
├── resnet_dcd/
│   └── resnet_dcd.py
├── LICENSE
└── README.md
```

## Acknowledgment

This implementation builds upon the original GLNet repository:

[GLNet](https://github.com/wuchangsheng951/GLNET)

## Citation

Please cite the following paper when using this code:

```bibtex
@inproceedings{akbaba2023improved,
  title={Improved Scene Classification by Dynamic CNNs},
  author={Akbaba, Elif Ecem and Gunsel, Bilge and Gurkan, Filiz},
  booktitle={2023 30th IEEE International Conference on Electronics, Circuits and Systems (ICECS)},
  pages={1--4},
  year={2023},
  organization={IEEE}
}
```
