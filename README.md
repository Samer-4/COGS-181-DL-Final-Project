# Multi-Disease Chest X-Ray Classification using Deep Learning

This project implements a deep learning solution for detecting multiple diseases from chest X-ray images using Convolutional Neural Networks (CNNs). The model is trained on the NIH Chest X-ray dataset and can detect multiple conditions including pneumonia, mass, and fibrosis.

## Project Structure
```
.
├── data/                    # Data directory (not tracked in git)
│   └── nih_chest_xray/     # NIH Chest X-ray dataset
├── src/                    # Source code
│   ├── models/            # Neural network architectures
│   ├── data/              # Data loading and preprocessing
│   ├── utils/             # Utility functions
│   └── train.py           # Training script
├── notebooks/             # Jupyter notebooks for analysis
├── configs/               # Configuration files
└── requirements.txt       # Project dependencies
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the NIH Chest X-ray dataset and place it in the `data/nih_chest_xray/` directory.

## Training

To train the model:
```bash
python src/train.py --config configs/baseline.yaml
```

## Features

- Multi-label classification for 14 different chest conditions
- Support for various CNN architectures (ResNet, DenseNet, EfficientNet)
- Extensive data augmentation pipeline
- Training with class weights to handle imbalanced data
- Experiment tracking with Weights & Biases
- Model interpretability using GradCAM

## Results

The model achieves the following performance metrics:
- Average AUC-ROC: TBD
- Average F1-Score: TBD
- Per-class metrics: TBD

## License

MIT License 