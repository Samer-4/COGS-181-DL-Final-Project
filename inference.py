import argparse

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image

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


def get_inference_transform():
    """Return the preprocessing used for validation/test images."""
    return A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )


def load_model(checkpoint_path, device):
    """Load the trained ResNet-50 checkpoint."""
    model = ChestXRayNet(
        num_classes=len(CONDITIONS),
        model_name="resnet50",
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


def predict(model, image_path, device):
    """Predict probabilities for all 14 disease labels."""
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)

    transform = get_inference_transform()

    image_tensor = transform(
        image=image
    )["image"]

    # Add batch dimension:
    # [3, 224, 224] -> [1, 3, 224, 224]
    image_tensor = (
        image_tensor
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        logits = model(image_tensor)

        probabilities = torch.sigmoid(
            logits
        )[0]

    probabilities = (
        probabilities
        .cpu()
        .numpy()
    )

    predictions = {
        condition: float(probability)
        for condition, probability
        in zip(CONDITIONS, probabilities)
    }

    return predictions


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run multi-label chest X-ray "
            "classification inference."
        )
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the chest X-ray image.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to best_model.pth.",
    )

    args = parser.parse_args()

    # Select the best available device.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    model = load_model(
        args.checkpoint,
        device,
    )

    predictions = predict(
        model,
        args.image,
        device,
    )

    print("\nPredicted probabilities:")
    print("-" * 40)

    # Display highest probabilities first.
    sorted_predictions = sorted(
        predictions.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    for condition, probability in sorted_predictions:
        print(
            f"{condition:<20} "
            f"{probability:.4f}"
        )


if __name__ == "__main__":
    main()
