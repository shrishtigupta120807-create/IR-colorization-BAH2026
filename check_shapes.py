import numpy as np

tir = np.load(
    "output/patches/scene1/sample_000/tir_100m_512.npy"
)

rgb = np.load(
    "output/patches/scene1/sample_000/rgb_100m_512.npy"
)

print("TIR Shape :", tir.shape)
print("RGB Shape :", rgb.shape)

print("\nTIR Min :", tir.min())
print("TIR Max :", tir.max())

print("\nRGB Min :", rgb.min())
print("RGB Max :", rgb.max())