import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        input_file, target_file = self.samples[idx]

        x = np.load(input_file).astype(np.float32) / 30000.0
        y = np.load(target_file).astype(np.float32) / 30000.0

        x = torch.tensor(x)
        y = torch.tensor(y)

        return x, y


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


dataset = TIRDataset("output/patches")

dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

model = SimpleSR()

loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

for epoch in range(10):

    total_loss = 0

    for x, y in dataloader:

        optimizer.zero_grad()

        prediction = model(x)

        loss = loss_fn(
            prediction,
            y
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch+1} Loss: "
        f"{total_loss:.4f}"
    )
torch.save(
    model.state_dict(),
    "sr_model.pth"
)

print("Model saved!")