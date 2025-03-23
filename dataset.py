import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ChestXRayDataset(Dataset):
    def __init__(self, data_dir, csv_file, transform=None, phase='train'):
        self.data_dir = data_dir
        self.df = pd.read_csv(csv_file)  
        self.df = self.df[self.df['Image Index'].apply(
            lambda x: os.path.exists(os.path.join(self.data_dir, x))
        )]
        self.df = self.df.reset_index(drop=True)
        
        self.phase = phase
        self.conditions = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
            'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
            'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]
        for condition in self.conditions:
            self.df[condition] = self.df['Finding Labels'].apply(
                lambda x: 1 if condition in x else 0
            )
        if transform:
            self.transform = transform
        else:
            self.transform = self._get_default_transforms()

    def _get_default_transforms(self):
        if self.phase == 'train':
            return A.Compose([
                A.Resize(224, 224),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.data_dir, self.df.iloc[idx]['Image Index'])
        
        try:
            image = Image.open(img_name).convert('RGB')
            image = np.array(image)
        except Exception as e:
            print(f"Warning: Failed to load image {img_name}. Skipping. Error: {e}")
            return self.__getitem__((idx + 1) % len(self.df))
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        labels = torch.FloatTensor(
            self.df.iloc[idx][self.conditions].values.astype(np.float32)
        )
        
        return {
            'image': image,
            'labels': labels,
            'image_path': img_name
        }

# Example usage:
# Since images are in the same folder as this Python file, we set data_dir to "."
data_dir = "."
csv_file = "Data_Entry_2017_v2020.csv"
dataset = ChestXRayDataset(data_dir=data_dir, csv_file=csv_file, phase='train')
