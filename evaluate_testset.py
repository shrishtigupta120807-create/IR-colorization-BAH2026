import torch
import torch.nn as nn
import numpy as np
import os

from torch.utils.data import Dataset
test_indices = torch.load(
    "test_indices.pth"
)

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)
class TIRDataset(Dataset):

    def __init__(self, patches_dir):

        self.samples = []

        for scene in os.listdir(patches_dir):

            if scene == "demo":
                continue

            scene_path = os.path.join(patches_dir, scene)

            if not os.path.isdir(scene_path):
                continue

            for sample in os.listdir(scene_path):

                sample_path = os.path.join(scene_path, sample)

                input_file = os.path.join(
                    sample_path,
                    "tir_200m.npy"
                )

                target_file = os.path.join(
                    sample_path,
                    "tir_100m_512.npy"
                )

                self.samples.append(
                    (input_file, target_file)
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
import os

psnr_scores = []
ssim_scores = []


dataset = TIRDataset("output/patches")

for idx in test_indices:

    input_file, target_file = dataset.samples[idx]

    input_img = np.load(
        input_file
    ).astype(np.float32) / 30000.0

    target_img = np.load(
        target_file
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

    psnr_scores.append(psnr)
    ssim_scores.append(ssim)



print("Total samples evaluated:", len(psnr_scores))

print(
    "Average PSNR =",
    np.mean(psnr_scores)
)

print(
    "Average SSIM =",
    np.mean(ssim_scores)
)