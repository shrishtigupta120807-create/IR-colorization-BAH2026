import torch
import torch.nn as nn


class SimpleSR(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Upsample(
                size=(512, 512),
                mode="bilinear"
            ),

            nn.Conv2d(
                1,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                16,
                1,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, x):

        return self.network(x)


model = SimpleSR()

model.load_state_dict(
    torch.load("sr_model.pth")
)

print("Model loaded successfully!")