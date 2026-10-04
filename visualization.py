import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


class GradCAM:
    """
    Generate Grad-CAM heatmaps for a target convolutional layer.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self.forward_handle = (
            self.target_layer.register_forward_hook(
                self._forward_hook
            )
        )

        self.backward_handle = (
            self.target_layer.register_full_backward_hook(
                self._backward_hook
            )
        )

    def _forward_hook(self, module, inputs, output):
        self.activations = output

    def _backward_hook(
        self,
        module,
        grad_input,
        grad_output,
    ):
        self.gradients = grad_output[0]

    def generate_cam(
        self,
        input_tensor,
        target_class=None,
    ):
        """
        Generate a normalized Grad-CAM heatmap.

        Args:
            input_tensor:
                Model input with shape [1, 3, H, W].

            target_class:
                Disease class index to explain. If None,
                the highest-logit class is selected.

        Returns:
            NumPy heatmap normalized to [0, 1].
        """
        self.model.eval()

        output = self.model(input_tensor)

        if target_class is None:
            target_class = (
                output[0]
                .argmax()
                .item()
            )

        self.model.zero_grad()

        score = output[0, target_class]
        score.backward()

        if self.gradients is None:
            raise RuntimeError(
                "Gradients were not captured."
            )

        if self.activations is None:
            raise RuntimeError(
                "Activations were not captured."
            )

        gradients = (
            self.gradients[0]
            .detach()
            .cpu()
            .numpy()
        )

        activations = (
            self.activations[0]
            .detach()
            .cpu()
            .numpy()
        )

        # Average gradients spatially to determine
        # the importance of each feature channel.
        weights = np.mean(
            gradients,
            axis=(1, 2),
        )

        cam = np.zeros(
            activations.shape[1:],
            dtype=np.float32,
        )

        for weight, activation in zip(
            weights,
            activations,
        ):
            cam += weight * activation

        # Standard Grad-CAM applies ReLU.
        cam = np.maximum(cam, 0)

        # Resize heatmap to the model input dimensions.
        height = input_tensor.shape[2]
        width = input_tensor.shape[3]

        cam = cv2.resize(
            cam,
            (width, height),
        )

        # Normalize to [0, 1].
        cam -= cam.min()

        max_value = cam.max()

        if max_value > 0:
            cam /= max_value

        return cam

    def close(self):
        """Remove registered PyTorch hooks."""
        self.forward_handle.remove()
        self.backward_handle.remove()


def visualize_prediction(
    image,
    cam,
    prediction,
    true_label=None,
    save_path=None,
):
    """
    Display an X-ray alongside its Grad-CAM heatmap.
    """

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 5),
    )

    axes[0].imshow(
        image,
        cmap="gray",
    )
    axes[0].set_title("Original X-ray")
    axes[0].axis("off")

    axes[1].imshow(
        image,
        cmap="gray",
    )
    axes[1].imshow(
        cam,
        cmap="jet",
        alpha=0.5,
    )
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")

    if true_label is not None:
        fig.suptitle(
            f"Prediction: {prediction}\n"
            f"True Label: {true_label}"
        )
    else:
        fig.suptitle(
            f"Prediction: {prediction}"
        )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(
            save_path,
            bbox_inches="tight",
            dpi=150,
        )
        plt.close(fig)

    else:
        plt.show()


def plot_training_history(
    history,
    save_path=None,
):
    """
    Plot training/validation loss and validation ROC-AUC.
    """

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4),
    )

    axes[0].plot(
        history["train_loss"],
        label="Training Loss",
    )

    axes[0].plot(
        history["val_loss"],
        label="Validation Loss",
    )

    axes[0].set_title("Loss History")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(
        history["val_auc"],
        label="Validation AUC",
    )

    axes[1].set_title(
        "Validation ROC-AUC"
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("ROC-AUC")
    axes[1].legend()

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(
            save_path,
            bbox_inches="tight",
            dpi=150,
        )
        plt.close(fig)

    else:
        plt.show()