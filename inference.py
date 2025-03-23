import torch
from model import ChestXRayNet

model = ChestXRayNet(num_classes=14, model_name='resnet50', pretrained=True)
model_path = "/Users/samerahmed/Desktop/COGS 181 DL Final Project/wandb/run-20250322_172358-cxxihgx4/files/best_model.pth"

model.load_state_dict(torch.load(model_path))
model.eval()

print("Model loaded successfully. Ready for inference!")
