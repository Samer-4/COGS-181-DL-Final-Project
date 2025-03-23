import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ChestXRayNet(nn.Module):
    def __init__(self, num_classes=14, model_name='resnet50', pretrained=True):
        super(ChestXRayNet, self).__init__()
        
        if model_name == 'resnet50':
            self.backbone = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity() 
        else:
            raise ValueError(f"Model {model_name} not supported")
        
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

    def get_attention_maps(self, x, class_idx=None):
        self.eval()
        feature_maps = []
        gradients = []

        def forward_hook(module, input, output):
            feature_maps.append(output)
        
        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])
        
        handle_forward = self.backbone.layer4[-1].conv3.register_forward_hook(forward_hook)
        handle_backward = self.backbone.layer4[-1].conv3.register_backward_hook(backward_hook)
        
        logits = self.forward(x) 
        
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        
        score = logits[0, class_idx]
        
        self.zero_grad()
        score.backward(retain_graph=True)
        
        handle_forward.remove()
        handle_backward.remove()
        fmap = feature_maps[0].detach()   
        grad = gradients[0].detach()       
        weights = grad.mean(dim=(2, 3), keepdim=True) 
        cam = (weights * fmap).sum(dim=1, keepdim=True)  
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = F.interpolate(cam, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        
        return cam