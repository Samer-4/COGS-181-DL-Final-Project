import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import wandb
from tqdm import tqdm
import pandas as pd

from model import ChestXRayNet
from dataset import ChestXRayDataset


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    pbar = tqdm(dataloader, desc="Training")

    for batch in pbar:
        images = batch["image"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})

    return running_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device, conditions):
    model.eval()
    running_loss = 0.0

    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch["image"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())

    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    probabilities = torch.sigmoid(all_outputs)

    auc_scores = {}

    for i, condition in enumerate(conditions):
        if len(torch.unique(all_labels[:, i])) > 1:
            auc = roc_auc_score(
                all_labels[:, i],
                probabilities[:, i],
            )
            auc_scores[condition] = auc

    average_auc = sum(auc_scores.values()) / len(auc_scores)

    return (
        running_loss / len(dataloader),
        average_auc,
        auc_scores,
    )


def calculate_pos_weights(csv_file, conditions, device):
    df = pd.read_csv(csv_file)

    weights = []

    for condition in conditions:
        positive = df["Finding Labels"].str.contains(
            condition,
            regex=False,
        ).sum()

        negative = len(df) - positive

        if positive == 0:
            raise ValueError(
                f"No positive samples found for {condition} in {csv_file}"
            )

        weight = negative / positive
        weights.append(weight)

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=device,
    )


def main(config):
    wandb.init(
        project="chest-xray-classification",
        config=config,
    )

    # Select the best available device.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load patient-level training and validation splits.
    train_dataset = ChestXRayDataset(
        data_dir=config["data_dir"],
        csv_file=config["train_csv"],
        phase="train",
    )

    val_dataset = ChestXRayDataset(
        data_dir=config["data_dir"],
        csv_file=config["val_csv"],
        phase="val",
    )

    if len(train_dataset) == 0:
        raise ValueError(
            "Train dataset is empty. Check your CSV file and image paths."
        )

    if len(val_dataset) == 0:
        raise ValueError(
            "Validation dataset is empty. Check your CSV file and image paths."
        )

    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )

    # Initialize the multi-label ResNet-50 classifier.
    model = ChestXRayNet(
        num_classes=config["num_classes"],
        model_name=config["model_name"],
        pretrained=config["pretrained"],
    ).to(device)

    conditions = [
        "Atelectasis",
        "Cardiomegaly",
        "Effusion",
        "Infiltration",
        "Mass",
        "Nodule",
        "Pneumonia",
        "Pneumothorax",
        "Consolidation",
        "Edema",
        "Emphysema",
        "Fibrosis",
        "Pleural_Thickening",
        "Hernia",
    ]

    # Weighted BCE compensates for severe class imbalance in ChestX-ray14.
    if config.get("use_class_weights", False):
        pos_weights = calculate_pos_weights(
            config["train_csv"],
            conditions,
            device,
        )

        print("Positive class weights:")

        for condition, weight in zip(conditions, pos_weights):
            print(f"{condition}: {weight.item():.2f}")

        criterion = nn.BCEWithLogitsLoss(
            pos_weight=pos_weights
        )

    else:
        print("Using unweighted BCE loss.")
        criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
    )

    best_val_auc = 0.0

    for epoch in range(config["num_epochs"]):
        print(
            f"\nEpoch {epoch + 1}/{config['num_epochs']}"
        )

        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss, val_auc, val_auc_scores = evaluate(
            model,
            val_loader,
            criterion,
            device,
            conditions,
        )

        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Average Validation AUC: {val_auc:.4f}")

        print("\nPer-class Validation AUC:")

        for condition, auc in val_auc_scores.items():
            print(f"  {condition}: {auc:.4f}")

        # Log training metrics to Weights & Biases.
        log_data = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "epoch": epoch + 1,
        }

        for condition, auc in val_auc_scores.items():
            log_data[f"val_auc/{condition}"] = auc

        wandb.log(log_data)

        # Select the checkpoint using validation ROC-AUC only.
        if val_auc > best_val_auc:
            best_val_auc = val_auc

            checkpoint_path = os.path.join(
                wandb.run.dir,
                "best_model.pth",
            )

            torch.save(
                model.state_dict(),
                checkpoint_path,
            )

            print(
                f"Saved new best model with "
                f"validation AUC: {val_auc:.4f}"
            )

    print(
        f"\nTraining complete. "
        f"Best validation AUC: {best_val_auc:.4f}"
    )

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file",
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    main(config)