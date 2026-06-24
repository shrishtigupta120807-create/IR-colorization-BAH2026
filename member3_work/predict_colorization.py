import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

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
     "output/patches/scene18/sample_010/tir_100m_512.npy"
).astype(np.float32)

# Normalize
tir_norm = tir /  56234.0

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
print("Shape:", pred.shape)
print("Min:", pred.min())
print("Max:", pred.max())

# Show image
plt.figure(figsize=(6,6))
plt.imshow(pred)
plt.title("Predicted RGB")
plt.axis("off")

plt.savefig("prediction.png")
print("Saved prediction.png")