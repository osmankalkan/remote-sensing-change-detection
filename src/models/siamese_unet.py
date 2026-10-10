"""Siamese U-Net with Shared-Weights Twin Encoder and Attention Fusion."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from .attention import CBAM


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU block."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class FusionBlock(nn.Module):
    """
    Fuses twin features f1 and f2:
    - Absolute difference: |f1 - f2|
    - Channel concat: [f1, f2, |f1 - f2|]
    - 1x1 Conv compression
    - CBAM attention filtering
    """
    def __init__(self, in_channels, out_channels, use_attention=True):
        super().__init__()
        # Concat has 3 * in_channels
        self.compress = nn.Sequential(
            nn.Conv2d(in_channels * 3, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.use_attention = use_attention
        if use_attention:
            self.attention = CBAM(out_channels)
        else:
            self.attention = nn.Identity()

    def forward(self, f1, f2):
        diff = torch.abs(f1 - f2)
        fused = torch.cat([f1, f2, diff], dim=1)
        compressed = self.compress(fused)
        return self.attention(compressed)


class DecoderBlock(nn.Module):
    """Upsampling followed by concatenation with skip feature and ConvBlock."""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SiameseUNet(nn.Module):
    """
    Siamese U-Net with Shared-Weights Twin Encoder:
    - Shared ResNet34/ResNet50 encoder extracts multi-scale features for T1 and T2.
    - Multi-scale fusion blocks aggregate features at each stage using difference + attention.
    - U-Net decoder synthesizes the change mask.
    """
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        classes: int = 1,
        use_attention: bool = True,
    ):
        super().__init__()
        self.encoder_name = encoder_name
        self.classes = classes

        # Shared Twin Encoder
        self.encoder = smp.encoders.get_encoder(
            name=encoder_name,
            in_channels=3,
            depth=5,
            weights=encoder_weights,
        )
        enc_channels = self.encoder.out_channels  # e.g., (3, 64, 64, 128, 256, 512)

        # Multi-scale fusion blocks for skip connections (stages 0 to 4)
        self.fusion_blocks = nn.ModuleList([
            FusionBlock(ch, ch, use_attention=use_attention) for ch in enc_channels
        ])

        # Decoder blocks
        # Bottleneck stage 5:
        self.center = ConvBlock(enc_channels[-1], enc_channels[-1])
        
        # Stages 4, 3, 2, 1, 0
        self.dec4 = DecoderBlock(enc_channels[5], enc_channels[4], enc_channels[4])
        self.dec3 = DecoderBlock(enc_channels[4], enc_channels[3], enc_channels[3])
        self.dec2 = DecoderBlock(enc_channels[3], enc_channels[2], enc_channels[2])
        self.dec1 = DecoderBlock(enc_channels[2], enc_channels[1], enc_channels[1])
        self.dec0 = DecoderBlock(enc_channels[1], enc_channels[0], 32)

        # Output head (1x1 Conv)
        self.head = nn.Conv2d(32, classes, kernel_size=1)

    def forward(self, t1, t2):
        """
        Args:
            t1: [B, 3, H, W] pre-event image
            t2: [B, 3, H, W] post-event image
            
        Returns:
            logits: [B, classes, H, W]
        """
        # Twin encoder pass (shared weights)
        feats1 = self.encoder(t1)
        feats2 = self.encoder(t2)

        # Fuse features at each stage
        fused_feats = [
            fusion(f1, f2)
            for fusion, f1, f2 in zip(self.fusion_blocks, feats1, feats2)
        ]

        # Decoder
        center = self.center(fused_feats[5])
        d4 = self.dec4(center, fused_feats[4])
        d3 = self.dec3(d4, fused_feats[3])
        d2 = self.dec2(d3, fused_feats[2])
        d1 = self.dec1(d2, fused_feats[1])
        d0 = self.dec0(d1, fused_feats[0])

        logits = self.head(d0)
        return logits
