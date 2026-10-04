import os

import pandas as pd
from sklearn.model_selection import train_test_split

CSV_FILE = "Data_Entry_2017_v2020.csv"
OUTPUT_DIR = "splits"
RANDOM_SEED = 42

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


def count_conditions(split_df):
    """Count positive examples for each disease label."""
    finding_sets = split_df["Finding Labels"].apply(
        lambda labels: set(labels.split("|"))
    )

    return {
        condition: finding_sets.apply(
            lambda labels: condition in labels
        ).sum()
        for condition in CONDITIONS
    }


def main():
    # Load the full NIH ChestX-ray14 metadata.
    # Splits are generated from patient IDs in the metadata,
    # independent of whether the image files are stored locally.
    df = pd.read_csv(CSV_FILE)

    print(f"Total images: {len(df):,}")
    print(
        f"Unique patients: "
        f"{df['Patient ID'].nunique():,}"
    )

    if len(df) == 0:
        raise ValueError(
            "Metadata file is empty."
        )

    # Split by patient rather than by image to prevent
    # patient leakage across train/validation/test sets.
    patient_ids = df["Patient ID"].unique()

    # 70% train, 30% temporary.
    train_patients, temp_patients = train_test_split(
        patient_ids,
        test_size=0.30,
        random_state=RANDOM_SEED,
    )

    # Divide the remaining 30% equally:
    # 15% validation and 15% test.
    val_patients, test_patients = train_test_split(
        temp_patients,
        test_size=0.50,
        random_state=RANDOM_SEED,
    )

    # Assign every X-ray from a patient to the same split.
    train_df = df[
        df["Patient ID"].isin(train_patients)
    ].copy()

    val_df = df[
        df["Patient ID"].isin(val_patients)
    ].copy()

    test_df = df[
        df["Patient ID"].isin(test_patients)
    ].copy()

    print("\n--- Split Summary ---")

    print(
        f"Train: {len(train_df)} images, "
        f"{train_df['Patient ID'].nunique()} patients"
    )

    print(
        f"Val:   {len(val_df)} images, "
        f"{val_df['Patient ID'].nunique()} patients"
    )

    print(
        f"Test:  {len(test_df)} images, "
        f"{test_df['Patient ID'].nunique()} patients"
    )

    # Verify that no patient appears in multiple splits.
    train_ids = set(train_df["Patient ID"])
    val_ids = set(val_df["Patient ID"])
    test_ids = set(test_df["Patient ID"])

    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = val_ids & test_ids

    print("\n--- Patient Overlap Check ---")
    print(
        f"Train / Val overlap:  "
        f"{len(train_val_overlap)}"
    )
    print(
        f"Train / Test overlap: "
        f"{len(train_test_overlap)}"
    )
    print(
        f"Val / Test overlap:   "
        f"{len(val_test_overlap)}"
    )

    # Fail immediately if patient leakage is detected.
    assert len(train_val_overlap) == 0
    assert len(train_test_overlap) == 0
    assert len(val_test_overlap) == 0

    # Report label distribution across splits.
    distribution = pd.DataFrame(
        {
            "Train": count_conditions(train_df),
            "Val": count_conditions(val_df),
            "Test": count_conditions(test_df),
        }
    )

    print("\n--- Label Distribution ---")
    print(distribution)

    # Save patient-level split metadata.
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    train_path = os.path.join(
        OUTPUT_DIR,
        "train.csv",
    )
    val_path = os.path.join(
        OUTPUT_DIR,
        "val.csv",
    )
    test_path = os.path.join(
        OUTPUT_DIR,
        "test.csv",
    )

    train_df.to_csv(
        train_path,
        index=False,
    )
    val_df.to_csv(
        val_path,
        index=False,
    )
    test_df.to_csv(
        test_path,
        index=False,
    )

    print("\nSaved:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {test_path}")


if __name__ == "__main__":
    main()