import os

import albumentations as A
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset


class ChestXRayDataset(Dataset):
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

    def __init__(
        self,
        data_dir,
        csv_file,
        transform=None,
        phase="train",
    ):
        self.data_dir = data_dir
        self.phase = phase

        # Load metadata / patient split.
        self.df = pd.read_csv(csv_file)
        self.total_csv_entries = len(self.df)

        # Keep only rows whose image exists in data_dir.
        image_exists = self.df["Image Index"].apply(
            lambda image_name: os.path.exists(
                os.path.join(self.data_dir, image_name)
            )
        )

        self.missing_images = int((~image_exists).sum())

        self.df = (
            self.df[image_exists]
            .reset_index(drop=True)
        )

        self.conditions = self.CONDITIONS

        # Convert the pipe-separated NIH labels into
        # 14 independent binary targets.
        finding_sets = self.df["Finding Labels"].apply(
            lambda labels: set(labels.split("|"))
        )

        for condition in self.conditions:
            self.df[condition] = finding_sets.apply(
                lambda labels: float(condition in labels)
            )

        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._get_default_transforms()

    def _get_default_transforms(self):
        if self.phase == "train":
            return A.Compose(
                [
                    A.Resize(224, 224),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.2),
                    A.ShiftScaleRotate(
                        shift_limit=0.05,
                        scale_limit=0.05,
                        rotate_limit=15,
                        p=0.5,
                    ),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                    ToTensorV2(),
                ]
            )

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

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        row = self.df.iloc[idx]

        image_path = os.path.join(
            self.data_dir,
            row["Image Index"],
        )

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load image: {image_path}"
            ) from exc

        image = np.array(image)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        labels = torch.tensor(
            row[self.conditions]
            .values.astype(np.float32),
            dtype=torch.float32,
        )

        return {
            "image": image,
            "labels": labels,
            "image_path": image_path,
        }


if __name__ == "__main__":
    data_dir = "images"
    csv_file = "Data_Entry_2017_v2020.csv"

    dataset = ChestXRayDataset(
        data_dir=data_dir,
        csv_file=csv_file,
        phase="train",
    )

    print("\n--- Dataset Summary ---")
    print(f"CSV entries: {dataset.total_csv_entries}")
    print(f"Images found: {len(dataset)}")
    print(f"Images missing: {dataset.missing_images}")
    print(
        f"Unique patients: "
        f"{dataset.df['Patient ID'].nunique()}"
    )

    print("\n--- Label Distribution ---")

    print(
        dataset.df[dataset.conditions]
        .sum()
        .sort_values(ascending=False)
    )

    no_finding = (
        dataset.df["Finding Labels"] == "No Finding"
    ).sum()

    print(f"\nNo Finding: {no_finding}")

    multi_label = (
        dataset.df["Finding Labels"]
        .str.split("|")
        .apply(len)
        .gt(1)
        .sum()
    )

    print(f"Multi-label images: {multi_label}")