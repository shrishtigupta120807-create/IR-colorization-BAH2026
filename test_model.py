import torch

from colorization_model import ColorizationUNet

model = ColorizationUNet()

x = torch.randn(1, 1, 512, 512)

y = model(x)

print("Input Shape :", x.shape)

print("Output Shape:", y.shape)
