import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ChestXRayNet(nn.Module):
    def __init__(
        self,
        num_classes=14,
        model_name="resnet50",
        pretrained=True,
    ):
        super().__init__()

        if model_name == "resnet50":
            weights = (
                models.ResNet50_Weights.IMAGENET1K_V2
                if pretrained
                else None
            )

            self.backbone = models.resnet50(
                weights=weights
            )

            num_features = self.backbone.fc.in_features

            # Remove the original ImageNet classifier.
            self.backbone.fc = nn.Identity()

        else:
            raise ValueError(
                f"Model '{model_name}' is not supported."
            )

        # Multi-label classification head.
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

        # Initialize the custom classification layers.
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)

        # Return raw logits.
        # BCEWithLogitsLoss applies sigmoid internally during training.
        return logits

    def get_attention_maps(self, x, class_idx=None):
        """
        Generate a Grad-CAM attention map for one class.

        Args:
            x:
                Input tensor with shape [1, 3, H, W].

            class_idx:
                Index of the disease class to visualize.
                If None, the class with the highest logit is used.

        Returns:
            Normalized Grad-CAM tensor with shape
            [1, 1, H, W].
        """
        self.eval()

        feature_maps = []
        gradients = []

        target_layer = self.backbone.layer4[-1].conv3

        def forward_hook(module, inputs, output):
            feature_maps.append(output)

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])

        forward_handle = target_layer.register_forward_hook(
            forward_hook
        )

        backward_handle = (
            target_layer.register_full_backward_hook(
                backward_hook
            )
        )

        try:
            logits = self.forward(x)

            if class_idx is None:
                class_idx = logits.argmax(dim=1).item()

            score = logits[0, class_idx]

            self.zero_grad()
            score.backward()

            feature_map = feature_maps[0].detach()
            gradient = gradients[0].detach()

            # Global-average-pool the gradients to obtain
            # the importance weight of each feature channel.
            weights = gradient.mean(
                dim=(2, 3),
                keepdim=True,
            )

            cam = (
                weights * feature_map
            ).sum(
                dim=1,
                keepdim=True,
            )

            cam = F.relu(cam)

            # Normalize Grad-CAM to [0, 1].
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)

            # Resize to the original model input resolution.
            cam = F.interpolate(
                cam,
                size=(x.size(2), x.size(3)),
                mode="bilinear",
                align_corners=False,
            )

            return cam

        finally:
            # Always remove hooks after Grad-CAM generation.
            forward_handle.remove()
            backward_handle.remove()