import torch
import torch.nn as nn
import numpy as np


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

model.eval()

sample = np.load(
    "output/patches/scene1/sample_000/tir_200m.npy"
).astype(np.float32)

sample = sample / 30000.0

sample = torch.tensor(sample)

sample = sample.unsqueeze(0)

with torch.no_grad():

    prediction = model(sample)

print("Input Shape :", sample.shape)
print("Prediction Shape :", prediction.shape)