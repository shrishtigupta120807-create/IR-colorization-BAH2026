import torch
import torch.nn as nn


class ColorizationUNet(nn.Module):

    def __init__(self):

        super().__init__()

        # Encoder

        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU()
        )

        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )

        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck

        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )

        # Decoder

        self.up1 = nn.ConvTranspose2d(
            128, 64, kernel_size=2, stride=2
        )

        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )

        self.up2 = nn.ConvTranspose2d(
            64, 32, kernel_size=2, stride=2
        )

        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU()
        )

        self.final = nn.Conv2d(
            32, 3, kernel_size=1
        )

    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool1(e1)
        )

        b = self.bottleneck(
            self.pool2(e2)
        )

        d1 = self.up1(b)

        d1 = torch.cat([d1, e2], dim=1)

        d1 = self.dec1(d1)

        d2 = self.up2(d1)

        d2 = torch.cat([d2, e1], dim=1)

        d2 = self.dec2(d2)

        out = self.final(d2)

        return out