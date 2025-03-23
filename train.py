import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score
import wandb
from tqdm import tqdm

from model import ChestXRayNet
from dataset import ChestXRayDataset

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    return running_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            images = batch['image'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())
    
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    auc_scores = []
    for i in range(all_outputs.shape[1]):
        if len(torch.unique(all_labels[:, i])) > 1:
            auc = roc_auc_score(all_labels[:, i], torch.sigmoid(all_outputs[:, i]))
            auc_scores.append(auc)
    
    return running_loss / len(dataloader), sum(auc_scores) / len(auc_scores)

def main(config):
    wandb.init(project="chest-xray-classification", config=config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = ChestXRayDataset(
        data_dir=config['data_dir'],
        csv_file=config['train_csv'],
        phase='train'
    )
    val_dataset = ChestXRayDataset(
        data_dir=config['data_dir'],
        csv_file=config['val_csv'],
        phase='val'
    )
    train_len = len(train_dataset)
    val_len = len(val_dataset)
    
    if train_len == 0:
        raise ValueError("Train dataset is empty. Check your CSV file and image paths.")
    if val_len == 0:
        raise ValueError("Validation dataset is empty. Check your CSV file and image paths.")
    
    train_subset_size = min(100, train_len)
    val_subset_size = min(100, val_len)
    
    train_dataset = Subset(train_dataset, range(train_subset_size))
    val_dataset = Subset(val_dataset, range(val_subset_size))
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    model = ChestXRayNet(
        num_classes=config['num_classes'],
        model_name=config['model_name'],
        pretrained=config['pretrained']
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    best_val_auc = 0.0
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)
        
        wandb.log({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_auc': val_auc,
            'epoch': epoch
        })
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'best_model.pth'))
            print(f"Saved new best model with validation AUC: {val_auc:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    main(config)
