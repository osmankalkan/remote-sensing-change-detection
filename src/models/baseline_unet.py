"""Early Fusion Baseline U-Net Model for Remote Sensing Change Detection."""
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class BaselineEarlyFusionUNet(nn.Module):
    """
    Early Fusion Baseline Model:
    Concatenates T1 (3 channels) and T2 (3 channels) along channel dimension
    into a 6-channel input tensor [B, 6, H, W], and passes through a standard
    U-Net with a pretrained backbone (e.g., ResNet18).
    """
    def __init__(
        self,
        encoder_name: str = "resnet18",
        encoder_weights: str = "imagenet",
        in_channels: int = 6,
        classes: int = 1,
        activation: str = None,
    ):
        super().__init__()
        self.encoder_name = encoder_name
        self.in_channels = in_channels
        self.classes = classes
        
        # Build SMP U-Net
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
        )

    def forward(self, t1, t2=None):
        """
        Forward pass.
        
        Args:
            t1: Tensor of shape [B, 3, H, W] if t2 is provided, 
                or [B, 6, H, W] if already concatenated.
            t2: Optional Tensor of shape [B, 3, H, W].
            
        Returns:
            logits: Tensor of shape [B, 1, H, W]
        """
        if t2 is not None:
            # Concatenate along channel dimension: [B, 3, H, W] + [B, 3, H, W] -> [B, 6, H, W]
            x = torch.cat([t1, t2], dim=1)
        else:
            x = t1
            
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected input with {self.in_channels} channels, but received {x.shape[1]} channels."
            )
            
        return self.unet(x)
