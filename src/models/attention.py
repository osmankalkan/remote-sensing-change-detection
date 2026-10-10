"""Attention mechanisms for Remote Sensing Change Detection (CBAM, Spatial & Channel Attention)."""
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (CAM) from CBAM.
    Applies both AvgPool and MaxPool, passes through shared MLP, and produces channel weights.
    """
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        reduced_channels = max(in_channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_pool = torch.mean(x, dim=(2, 3))
        max_pool = torch.amax(x, dim=(2, 3))

        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)

        channel_weights = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * channel_weights


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (SAM) from CBAM.
    Aggregates channel information using AvgPool & MaxPool, then applies 7x7 Conv.
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel_size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.amax(x, dim=1, keepdim=True)
        feat = torch.cat([avg_out, max_out], dim=1)
        spatial_weights = self.sigmoid(self.conv(feat))
        return x * spatial_weights


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Sequentially applies Channel Attention and Spatial Attention.
    """
    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction=reduction)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x
