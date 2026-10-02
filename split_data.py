import os
import pandas as pd
from sklearn.model_selection import train_test_split


DATA_DIR = "images"
CSV_FILE = "Data_Entry_2017_v2020.csv"
RANDOM_SEED = 42


# Load the full metadata
df = pd.read_csv(CSV_FILE)

# Keep only X-rays that actually exist on this computer
image_exists = df["Image Index"].apply(
    lambda x: os.path.exists(os.path.join(DATA_DIR, x))
)

df = df[image_exists].reset_index(drop=True)

print(f"Available images: {len(df)}")
print(f"Unique patients: {df['Patient ID'].nunique()}")


# Get one copy of every patient ID
patient_ids = df["Patient ID"].unique()


# 70% train, 30% temporary
train_patients, temp_patients = train_test_split(
    patient_ids,
    test_size=0.30,
    random_state=RANDOM_SEED,
)


# Split remaining 30% equally:
# 15% validation, 15% test
val_patients, test_patients = train_test_split(
    temp_patients,
    test_size=0.50,
    random_state=RANDOM_SEED,
)


# Give each X-ray to the split containing its patient
train_df = df[df["Patient ID"].isin(train_patients)]
val_df = df[df["Patient ID"].isin(val_patients)]
test_df = df[df["Patient ID"].isin(test_patients)]


print("\n--- Split Summary ---")
print(f"Train: {len(train_df)} images, {train_df['Patient ID'].nunique()} patients")
print(f"Val:   {len(val_df)} images, {val_df['Patient ID'].nunique()} patients")
print(f"Test:  {len(test_df)} images, {test_df['Patient ID'].nunique()} patients")

# Check for patient leakage
train_ids = set(train_df["Patient ID"])
val_ids = set(val_df["Patient ID"])
test_ids = set(test_df["Patient ID"])

print("\n--- Patient Overlap Check ---")
print(f"Train / Val overlap:  {len(train_ids & val_ids)}")
print(f"Train / Test overlap: {len(train_ids & test_ids)}")
print(f"Val / Test overlap:   {len(val_ids & test_ids)}")

conditions = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]


def count_conditions(split_df):
    counts = {}

    for condition in conditions:
        counts[condition] = split_df["Finding Labels"].str.contains(
            condition,
            regex=False
        ).sum()

    return counts


train_counts = count_conditions(train_df)
val_counts = count_conditions(val_df)
test_counts = count_conditions(test_df)

distribution = pd.DataFrame({
    "Train": train_counts,
    "Val": val_counts,
    "Test": test_counts
})

print("\n--- Label Distribution ---")
print(distribution)

os.makedirs("splits", exist_ok=True)

train_df.to_csv("splits/train.csv", index=False)
val_df.to_csv("splits/val.csv", index=False)
test_df.to_csv("splits/test.csv", index=False)

print("\nSaved:")
print("  splits/train.csv")
print("  splits/val.csv")
print("  splits/test.csv")