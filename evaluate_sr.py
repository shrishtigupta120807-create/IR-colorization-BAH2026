import torch
import torch.nn as nn
import numpy as np

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)


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


input_img = np.load(
    "output/patches/scene19/sample_015/tir_200m.npy"
).astype(np.float32) / 30000.0

target_img = np.load(
    "output/patches/scene19/sample_015/tir_100m_512.npy"
).astype(np.float32) / 30000.0


x = torch.tensor(
    input_img,
    dtype=torch.float32
).unsqueeze(0)

with torch.no_grad():
   

    prediction = model(x)

prediction = prediction.squeeze().numpy()

target = target_img.squeeze()


psnr = peak_signal_noise_ratio(
    target,
    prediction,
    data_range=1.0
)

ssim = structural_similarity(
    target,
    prediction,
    data_range=1.0
)

print("PSNR =", psnr)
print("SSIM =", ssim)