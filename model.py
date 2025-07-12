import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """Two stacked convolutional layers with BatchNorm and ReLU."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    """U-Net architecture for high-quality image reconstruction (used as teacher)."""
    def __init__(self):
        super().__init__()
        self.encoder1 = DoubleConv(3, 64)
        self.encoder2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.encoder3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.encoder4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))

        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(256, 128)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(128, 64)

        self.output_conv = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)

        x = self.upconv1(x4)
        x = self.decoder1(torch.cat([x, x3], dim=1))
        x = self.upconv2(x)
        x = self.decoder2(torch.cat([x, x2], dim=1))
        x = self.upconv3(x)
        x = self.decoder3(torch.cat([x, x1], dim=1))
        x = self.output_conv(x)

        return torch.clamp(x, 0, 1)

class StudentNet(nn.Module):
    """Simplified CNN for fast real-time image sharpening (used as student)."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.output_conv = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.output_conv(x)

        x = torch.sigmoid(x)
        enhanced = identity + (x - identity) * 0.3

        return torch.clamp(enhanced, 0, 1)
