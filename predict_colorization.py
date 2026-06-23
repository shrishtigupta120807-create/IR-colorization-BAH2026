import torch
import numpy as np
import matplotlib.pyplot as plt

from colorization_model import ColorizationUNet

# Load model
model = ColorizationUNet()

model.load_state_dict(
    torch.load(
        "colorization_model.pth",
        map_location="cpu"
    )
)

model.eval()

# Load one thermal image
tir = np.load(
    "output/patches/scene1/sample_000/tir_100m_512.npy"
).astype(np.float32)

# Normalize
tir_norm = tir / 27502.0

# Convert to tensor
tir_tensor = torch.tensor(
    tir_norm
).unsqueeze(0)

# Prediction
with torch.no_grad():

    pred = model(tir_tensor)

# Convert back to numpy
pred = pred.squeeze(0).numpy()

# Change shape from (3,H,W) to (H,W,3)
pred = np.transpose(pred, (1, 2, 0))

# Normalize for display
pred = (pred - pred.min()) / (
    pred.max() - pred.min()
)

# Show image
plt.imshow(pred)
plt.title("Predicted RGB")
plt.show()