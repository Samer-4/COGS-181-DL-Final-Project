import argparse
import json
import os

import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ChestXRayDataset
from model import ChestXRayNet


CONDITIONS = [
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


def evaluate(model, dataloader, criterion, device):
    """Evaluate a trained model on the held-out test set."""

    model.eval()

    running_loss = 0.0
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(
            dataloader,
            desc="Evaluating test set",
        ):
            images = batch["image"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            running_loss += loss.item()

            all_outputs.append(
                outputs.cpu()
            )

            all_labels.append(
                labels.cpu()
            )

    all_outputs = torch.cat(
        all_outputs,
        dim=0,
    )

    all_labels = torch.cat(
        all_labels,
        dim=0,
    )

    probabilities = torch.sigmoid(
        all_outputs
    )

    auc_scores = {}

    for i, condition in enumerate(CONDITIONS):
        labels_for_condition = all_labels[:, i]

        # ROC-AUC requires both positive and negative
        # examples to be present.
        if len(torch.unique(labels_for_condition)) > 1:
            auc = roc_auc_score(
                labels_for_condition.numpy(),
                probabilities[:, i].numpy(),
            )

            auc_scores[condition] = float(auc)

    average_auc = (
        sum(auc_scores.values())
        / len(auc_scores)
    )

    average_loss = (
        running_loss
        / len(dataloader)
    )

    return (
        average_loss,
        average_auc,
        auc_scores,
    )


def load_model(
    checkpoint_path,
    config,
    device,
):
    """Load the trained model checkpoint."""

    model = ChestXRayNet(
        num_classes=config["num_classes"],
        model_name=config["model_name"],
        pretrained=False,
    )

    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    return model


def main(args):
    # Load experiment configuration.
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # Select the best available device.
    if torch.cuda.is_available():
        device = torch.device("cuda")

    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    if device.type == "cuda":
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # Load the untouched patient-level test split.
    test_dataset = ChestXRayDataset(
        data_dir=config["data_dir"],
        csv_file=config["test_csv"],
        phase="test",
    )

    if len(test_dataset) == 0:
        raise ValueError(
            "Test dataset is empty. "
            "Check the image and CSV paths."
        )

    print(
        f"Test images: "
        f"{len(test_dataset)}"
    )

    print(
        f"Test patients: "
        f"{test_dataset.df['Patient ID'].nunique()}"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )

    model = load_model(
        args.checkpoint,
        config,
        device,
    )

    print("Model loaded successfully.")

    # We use ordinary BCE only to report test loss.
    #
    # ROC-AUC is computed directly from model
    # predictions and is unaffected by this criterion.
    criterion = nn.BCEWithLogitsLoss()

    test_loss, test_auc, per_class_auc = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    print("\n==============================")
    print("FINAL TEST RESULTS")
    print("==============================")

    print(
        f"Test Loss: "
        f"{test_loss:.4f}"
    )

    print(
        f"Average Test ROC-AUC: "
        f"{test_auc:.4f}"
    )

    print(
        "\nPer-class Test ROC-AUC:"
    )

    for condition, auc in per_class_auc.items():
        print(
            f"{condition:<20} "
            f"{auc:.4f}"
        )

    # Save reproducible test metrics.
    results = {
        "checkpoint": os.path.basename(
            args.checkpoint
        ),
        "test_images": len(test_dataset),
        "test_patients": int(
            test_dataset.df[
                "Patient ID"
            ].nunique()
        ),
        "average_test_roc_auc": float(
            test_auc
        ),
        "per_class_test_roc_auc": (
            per_class_auc
        ),
    }

    output_directory = os.path.dirname(
        args.output
    )

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True,
        )

    with open(
        args.output,
        "w",
    ) as file:
        json.dump(
            results,
            file,
            indent=4,
        )

    print(
        f"\nMetrics saved to: "
        f"{args.output}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained chest X-ray "
            "classifier on the held-out test set."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default="baseline.yaml",
        help="Path to experiment configuration.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to best_model.pth.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results/test_metrics.json",
        help="Path for saved test metrics.",
    )

    args = parser.parse_args()

    main(args)