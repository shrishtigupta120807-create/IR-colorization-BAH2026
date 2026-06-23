import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from colorization_dataset import ColorizationDataset
from colorization_model import ColorizationUNet

device = "cpu"

dataset = ColorizationDataset()

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True
)

model = ColorizationUNet().to(device)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 20

for epoch in range(epochs):

    total_loss = 0

    for tir, rgb in loader:

        tir = tir.float().to(device)
        rgb = rgb.float().to(device)

        tir = tir / 27502.0
        rgb = rgb / 11489.0

        pred = model(tir)

        loss = criterion(pred, rgb)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Loss: {total_loss/len(loader):.6f}"
    )

torch.save(
    model.state_dict(),
    "colorization_model.pth"
)

print("\nModel Saved!")