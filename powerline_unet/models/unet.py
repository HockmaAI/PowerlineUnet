"""
Attention U-Net model for power line segmentation.

This implements a U-Net with attention gates to better focus on fine power line structures.
The model was designed as part of an honors thesis for automated power line inspection.
"""
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    """Attention gate to focus on relevant features from skip connections."""
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Apply attention to skip connection.
        
        Args:
            g: Gating signal from decoder (upsampled)
            x: Skip connection from encoder
            
        Returns:
            Attended features from skip connection
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class UNet(nn.Module):
    """
    U-Net with attention gates for power line segmentation.
    
    Architecture:
    - 4-layer encoder with increasing channels
    - Bottleneck with 1024 channels
    - Decoder with attention gates on skip connections
    - Designed to handle class imbalance in power line detection
    """
    
    def __init__(
        self, 
        in_channels: int = 3, 
        out_channels: int = 1,
        encoder_channels: Optional[List[int]] = None,
        use_attention: bool = True
    ):
        super().__init__()
        
        if encoder_channels is None:
            encoder_channels = [64, 128, 256, 512]
            
        self.use_attention = use_attention
        
        def conv_block(in_c: int, out_c: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        # Encoder
        self.enc1 = conv_block(in_channels, encoder_channels[0])
        self.enc2 = conv_block(encoder_channels[0], encoder_channels[1])
        self.enc3 = conv_block(encoder_channels[1], encoder_channels[2])
        self.enc4 = conv_block(encoder_channels[2], encoder_channels[3])

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = conv_block(encoder_channels[3], 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, encoder_channels[3], kernel_size=2, stride=2)
        self.dec4 = conv_block(encoder_channels[3] * 2, encoder_channels[3])
        
        self.up3 = nn.ConvTranspose2d(encoder_channels[3], encoder_channels[2], kernel_size=2, stride=2)
        self.dec3 = conv_block(encoder_channels[2] * 2, encoder_channels[2])
        
        self.up2 = nn.ConvTranspose2d(encoder_channels[2], encoder_channels[1], kernel_size=2, stride=2)
        self.dec2 = conv_block(encoder_channels[1] * 2, encoder_channels[1])
        
        self.up1 = nn.ConvTranspose2d(encoder_channels[1], encoder_channels[0], kernel_size=2, stride=2)
        self.dec1 = conv_block(encoder_channels[0] * 2, encoder_channels[0])

        # Attention gates
        if self.use_attention:
            self.attention4 = AttentionGate(
                F_g=encoder_channels[3], F_l=encoder_channels[3], F_int=encoder_channels[2]
            )
            self.attention3 = AttentionGate(
                F_g=encoder_channels[2], F_l=encoder_channels[2], F_int=encoder_channels[1]
            )
            self.attention2 = AttentionGate(
                F_g=encoder_channels[1], F_l=encoder_channels[1], F_int=encoder_channels[0]
            )
            self.attention1 = AttentionGate(
                F_g=encoder_channels[0], F_l=encoder_channels[0], F_int=32
            )

        # Output layer
        self.out_conv = nn.Conv2d(encoder_channels[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional attention gates."""
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with optional attention
        d4 = self.up4(b)
        if self.use_attention:
            e4_att = self.attention4(d4, e4)
            d4 = self.dec4(torch.cat([d4, e4_att], dim=1))
        else:
            d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        if self.use_attention:
            e3_att = self.attention3(d3, e3)
            d3 = self.dec3(torch.cat([d3, e3_att], dim=1))
        else:
            d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        if self.use_attention:
            e2_att = self.attention2(d2, e2)
            d2 = self.dec2(torch.cat([d2, e2_att], dim=1))
        else:
            d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        if self.use_attention:
            e1_att = self.attention1(d1, e1)
            d1 = self.dec1(torch.cat([d1, e1_att], dim=1))
        else:
            d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)


# Convenience function
def create_model(config=None) -> UNet:
    """Create UNet model from config or with defaults."""
    if config is None:
        return UNet()
    return UNet(
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        encoder_channels=config.model.encoder_channels,
        use_attention=config.model.use_attention
    )
