import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from colorization_dataset import ColorizationDataset
from colorization_model import ColorizationUNet

device = "cuda"
best_loss = float("inf")

dataset = ColorizationDataset()

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

model = ColorizationUNet().to(device)

criterion = nn.L1Loss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 100
 
for epoch in range(epochs):

    total_loss = 0

    for tir, rgb in loader:

        tir = tir.float().to(device)
        rgb = rgb.float().to(device)

        tir = tir / 56234.0
        rgb = rgb / 65535.0

        pred = model(tir)

        loss = criterion(pred, rgb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # INSIDE epoch loop
    avg_loss = total_loss / len(loader)

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Loss: {avg_loss:.6f}"
    )

    if avg_loss < best_loss:

        best_loss = avg_loss

        torch.save(
            model.state_dict(),
            "colorization_model.pth"
        )

        print("Best model saved!") 

      
print("\nTraining Complete!")
print("Best Loss:", best_loss)        
    

