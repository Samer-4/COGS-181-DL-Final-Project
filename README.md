# COGS-181-DL-Final-Project

**Team Members:**  
- Samer Ahmed  
- Bassam Malik

## Overview

This project focuses on the automated diagnosis of thoracic diseases from chest X-ray images using deep learning. We developed a multi-label classification system capable of detecting 14 common conditions—such as pneumonia, cardiomegaly, and pulmonary effusion—using a convolutional neural network (CNN) based on the ResNet50 architecture. Due to the massive scale of the full NIH Chest X-ray dataset, we worked with a filtered subset of images, ensuring that only valid entries (with existing image files) were used. To enhance model generalization, we applied on-the-fly data augmentation with Albumentations and integrated Grad-CAM visualizations for interpretability. This README outlines our project structure, setup instructions, usage, and future directions.

## Repository Structure

```
COGS-181-DL-Final-Project/
├── README.md                  # This file
├── baseline.yaml              # Configuration file with experiment settings
├── Data_Entry_2017_v2020.csv   # CSV file with image metadata and labels
├── dataset.py                 # Custom dataset class for data loading and preprocessing
├── inference.py               # Script for running inference on new images
├── model.py                   # Model architecture (ResNet50-based) and Grad-CAM implementation
├── requirements.txt           # List of required Python packages
├── train.py                   # Training script to train the model and log metrics to wandb
├── visualization.py           # Utilities for generating visualizations (Grad-CAM, training history)
├── images/                    # Folder containing chest X-ray images
└── wandb/                     # Wandb logs and run artifacts (usually excluded via .gitignore)
```

## Setup Instructions

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Samer-4/COGS-181-DL-Final-Project.git
   cd COGS-181-DL-Final-Project
   ```

2. **Install Dependencies:**

   Ensure you have Python installed (using Anaconda is recommended), then run:

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Your Environment:**

   - If you’re using Anaconda, activate your environment before running any commands.
   - Make sure your images are in the `images` folder and the CSV file (`Data_Entry_2017_v2020.csv`) is in the repository root or update the paths accordingly.

## Usage

### Training the Model

The training process is controlled by the `train.py` script. Our configuration in `baseline.yaml` sets up the following:
- Data directory: `images`
- CSV file: `Data_Entry_2017_v2020.csv`
- Model: ResNet50 with custom classifier head
- Batch size: 32, Number of workers: 4
- Limited to the first 100 images for quick testing
- Number of epochs: 3
- Learning rate: 0.0001

To start training, run:

```bash
/opt/anaconda3/bin/python3 train.py --config baseline.yaml
```

During training, metrics (training loss, validation loss, and ROC-AUC) are logged to Weights & Biases (wandb), and the best model (based on validation AUC) is saved in the wandb run directory.

### Running Inference

After training, you can run the `inference.py` script to load the best saved model and make predictions on new chest X-ray images. Update the image path in `inference.py` as needed. To run inference:

```bash
/opt/anaconda3/bin/python3 inference.py
```

This script will load the model from the wandb run’s `files` folder, process the image, and print the predicted probabilities for each of the 14 conditions.

### Generating Visualizations

The `visualization.py` file provides functions to create Grad-CAM heatmaps and plot training history, helping you visually interpret the model's decisions and track its performance over time.

## Project Report

A detailed final report accompanies this repository. The report includes:
- **Abstract:** Summary of objectives, methodology, and key findings.
- **Introduction:** Background on chest X-ray diagnosis and the motivation for automation.
- **Method:** Detailed description of data preprocessing, model architecture, training procedures, and Grad-CAM integration.
- **Experiments:** Explanation of the experimental setup, hyperparameter tuning, results (quantitative metrics and qualitative Grad-CAM visualizations), and discussion of limitations.
- **Conclusion:** Insights from our work and potential future research directions.
- **References:** A list of key papers and resources that informed our approach.

## Future Work

Future improvements will focus on scaling up the data usage, refining the model architecture (exploring compact networks or transformer-based models), and implementing more robust evaluation techniques (such as proper train/validation/test splits or cross-validation). Additionally, further integration of interpretability methods could provide deeper insights into the model’s diagnostic process.
