import torch
from torch import nn

# Define Attention Gate
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
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

    def forward(self, g, x):
        # g: gating signal (from decoder, e.g., upsampled bottleneck)
        # x: skip connection (from encoder)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)  # Element-wise addition
        psi = self.psi(psi)       # Sigmoid activation for attention weights
        return x * psi            # Apply attention weights to skip connection

# Define U-Net Model with Attention
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        # Encoder
        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = conv_block(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = conv_block(1024, 512)  # Concatenates with attention output
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64)

        # Attention gates for each skip connection
        self.attention4 = AttentionGate(F_g=512, F_l=512, F_int=256)  # For enc4 -> dec4
        self.attention3 = AttentionGate(F_g=256, F_l=256, F_int=128)  # For enc3 -> dec3
        self.attention2 = AttentionGate(F_g=128, F_l=128, F_int=64)   # For enc2 -> dec2
        self.attention1 = AttentionGate(F_g=64, F_l=64, F_int=32)     # For enc1 -> dec1

        # Output
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder path with attention
        d4 = self.up4(b)
        e4_att = self.attention4(d4, e4)  # Apply attention to encoder feature
        d4 = self.dec4(torch.cat([d4, e4_att], dim=1))  # Concatenate with attended features

        d3 = self.up3(d4)
        e3_att = self.attention3(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3_att], dim=1))

        d2 = self.up2(d3)
        e2_att = self.attention2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2_att], dim=1))

        d1 = self.up1(d2)
        e1_att = self.attention1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1_att], dim=1))

        return self.out_conv(d1)  # No sigmoid, let the loss function handle it

# # Example usage (optional)
# if __name__ == "__main__":
#     # Create a sample input tensor (batch_size, channels, height, width)
#     sample_input = torch.randn(1, 3, 512, 512)
#     model = UNet(in_channels=3, out_channels=1)
#     output = model(sample_input)
#     print(f"Output shape: {output.shape}")  # Should be [1, 1, 512, 512]
