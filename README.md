# Multi-Label Chest X-Ray Classifier

**Deep learning pipeline for detecting 14 thoracic conditions from chest X-rays using PyTorch and ResNet-50.**

Developed by **Samer Ahmed**

## Overview

This project implements an end-to-end deep learning pipeline for **multi-label chest X-ray classification** using the NIH ChestX-ray14 dataset.

The model uses an ImageNet-pretrained **ResNet-50** backbone with a custom classification head to predict 14 thoracic conditions simultaneously. The full dataset contains **112,120 frontal chest X-rays from 30,805 patients**.

To produce a more reliable evaluation, the dataset is split at the **patient level** rather than the image level. This prevents X-rays belonging to the same patient from appearing across training, validation, and test sets.

The final model achieved a **mean ROC-AUC of 0.8225 on a held-out test set of 16,491 X-rays from 4,621 patients**.

The project includes:

- Patient-level train/validation/test splitting
- Multi-label classification across 14 thoracic conditions
- ImageNet-pretrained ResNet-50 transfer learning
- Class-weighted binary cross-entropy for severe label imbalance
- Albumentations-based image augmentation
- Per-class and mean ROC-AUC evaluation
- Weights & Biases experiment tracking
- Standalone inference and evaluation pipelines
- Grad-CAM visualizations for model interpretability

---

## Results

The best model checkpoint was selected using validation ROC-AUC and then evaluated once on the held-out patient-level test set.

| Metric | Result |
|---|---:|
| Mean Test ROC-AUC | **0.8225** |
| Test Images | **16,491** |
| Test Patients | **4,621** |
| Conditions | **14** |

### Per-Class Test ROC-AUC

| Condition | ROC-AUC |
|---|---:|
| Emphysema | **0.9268** |
| Cardiomegaly | **0.8962** |
| Hernia | **0.8949** |
| Edema | **0.8926** |
| Pneumothorax | **0.8775** |
| Effusion | **0.8719** |
| Mass | **0.8045** |
| Fibrosis | **0.7957** |
| Pleural Thickening | **0.7947** |
| Atelectasis | **0.7935** |
| Consolidation | **0.7913** |
| Nodule | **0.7422** |
| Pneumonia | **0.7380** |
| Infiltration | **0.6960** |

Performance varies substantially by condition, reflecting the difficulty and class imbalance of the dataset. The complete evaluation results are stored in `results/test_metrics.json`.

---

## Dataset

This project uses the **NIH ChestX-ray14** dataset.

The full metadata contains:

- **112,120 chest X-rays**
- **30,805 patients**
- **14 thoracic disease labels**
- Images with no positive disease labels represented as all-zero multi-label targets

The predicted conditions are:

`Atelectasis`, `Cardiomegaly`, `Effusion`, `Infiltration`, `Mass`, `Nodule`, `Pneumonia`, `Pneumothorax`, `Consolidation`, `Edema`, `Emphysema`, `Fibrosis`, `Pleural_Thickening`, and `Hernia`.

Because this is a **multi-label** problem, a single X-ray may contain more than one condition.

### Patient-Level Splitting

Rather than randomly splitting individual images, unique patient IDs are divided first and all X-rays belonging to a patient are assigned to the same partition.

| Split | Images | Patients |
|---|---:|---:|
| Training | 78,566 | 21,563 |
| Validation | 17,063 | 4,621 |
| Test | 16,491 | 4,621 |
| **Total** | **112,120** | **30,805** |

Patient overlap between all three partitions is **zero**.

The splits can be reproduced with:

```bash
python split_data.py
```

using random seed `42`.

The X-ray images themselves are **not included in this repository** due to dataset size. Download the NIH ChestX-ray14 dataset separately and place the extracted images inside an `images/` directory.

---

## Model Architecture

The classifier is built using an **ImageNet-pretrained ResNet-50**.

The original ResNet classification layer is removed and replaced with a custom multi-label classification head:

```text
Chest X-ray (224 × 224 × 3)
        │
        ▼
ImageNet-pretrained ResNet-50
        │
        ▼
2048-dimensional feature vector
        │
        ▼
Linear(2048 → 512)
        │
       ReLU
        │
   Dropout(0.3)
        │
        ▼
Linear(512 → 14)
        │
        ▼
14 disease logits
```

Raw logits are used during training with `BCEWithLogitsLoss`. Sigmoid is applied during inference to convert the logits into per-condition prediction scores.

---

## Training

The final experiment used:

| Parameter | Value |
|---|---|
| Backbone | ResNet-50 |
| Initialization | ImageNet pretrained |
| Input resolution | 224 × 224 |
| Batch size | 32 |
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Epochs | 10 |
| Loss | Weighted BCEWithLogitsLoss |
| Model selection | Validation ROC-AUC |
| Best checkpoint | Epoch 7 |
| Experiment tracking | Weights & Biases |

### Class Imbalance

ChestX-ray14 is highly imbalanced. Rare conditions such as Hernia and Pneumonia contain far fewer positive examples than common findings such as Infiltration.

Positive class weights are therefore calculated from the training split and supplied to `BCEWithLogitsLoss`, increasing the contribution of underrepresented positive examples during optimization.

### Data Augmentation

Training images use on-the-fly Albumentations transforms including:

- Resize to 224 × 224
- Horizontal flipping
- Random brightness and contrast
- Small translations, scaling, and rotations
- ImageNet normalization

Validation, test, and inference images use deterministic resizing and ImageNet normalization without random augmentation.

---

## Repository Structure

```text
.
├── README.md
├── baseline.yaml
├── Data_Entry_2017_v2020.csv
├── dataset.py
├── evaluate.py
├── inference.py
├── model.py
├── requirements.txt
├── split_data.py
├── train.py
├── visualization.py
│
├── splits/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
│
└── results/
    └── test_metrics.json
```

Large artifacts such as the NIH X-ray images, model checkpoints, and local Weights & Biases runs are intentionally excluded from Git.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Samer-4/COGS-181-DL-Final-Project.git
cd COGS-181-DL-Final-Project
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training

Ensure the NIH images are available under:

```text
images/
```

and then run:

```bash
python train.py --config baseline.yaml
```

Training metrics are logged to Weights & Biases.

The training pipeline tracks:

- Training loss
- Validation loss
- Mean validation ROC-AUC
- Per-class validation ROC-AUC

The checkpoint with the highest validation ROC-AUC is saved as `best_model.pth`.

---

## Evaluation

Evaluate a trained checkpoint against the held-out test set:

```bash
python evaluate.py \
    --config baseline.yaml \
    --checkpoint best_model.pth
```

The evaluation pipeline calculates:

- Mean test ROC-AUC
- ROC-AUC for each of the 14 conditions
- Test loss

Results are saved by default to:

```text
results/test_metrics.json
```

The test set remains separate from model training and checkpoint selection.

---

## Inference

Run inference on an individual chest X-ray:

```bash
python inference.py \
    --image path/to/xray.png \
    --checkpoint best_model.pth
```

The script preprocesses the X-ray using the same normalization used during validation and testing and outputs prediction scores for all 14 conditions.

These scores should not be interpreted as calibrated clinical probabilities or diagnoses.

---

## Grad-CAM Interpretability

The project includes **Grad-CAM** support for examining which spatial regions of an X-ray contribute to a selected model output.

Grad-CAM uses activations and gradients from the final convolutional portion of ResNet-50 to construct a heatmap that can be overlaid on the original X-ray.

This provides a qualitative way to inspect model attention, but the resulting heatmaps should **not** be interpreted as verified lesion localization.

---

## Technical Stack

**Deep Learning:** PyTorch, Torchvision
**Computer Vision:** Albumentations, OpenCV, Pillow
**Data & Evaluation:** Pandas, NumPy, scikit-learn
**Experiment Tracking:** Weights & Biases
**Visualization:** Matplotlib, Grad-CAM

---

## Limitations

This model should be interpreted as a machine-learning research project rather than a clinical system.

Important limitations include:

- Training and evaluation use a single public chest X-ray dataset.
- Dataset labels are imperfect and may contain noise.
- Performance varies substantially across conditions.
- ROC-AUC does not determine an appropriate clinical decision threshold.
- Prediction scores are not calibrated clinical probabilities.
- Grad-CAM provides qualitative model interpretation rather than validated pathology localization.
- No external clinical dataset has been used to evaluate generalization.

External validation, calibration, subgroup analysis, threshold selection, and prospective clinical evaluation would be required before considering any real-world medical use.

---

## Author

**Samer Ahmed**

MS Artificial Intelligence
Northeastern University

Interests: machine learning, AI engineering, and healthcare AI.

---
