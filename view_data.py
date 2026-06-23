import numpy as np
import matplotlib.pyplot as plt

# Load files
tir = np.load(
    "output/patches/demo/sample_006/tir_100m_512.npy"
)

rgb = np.load(
    "output/patches/demo/sample_006/rgb_100m_512.npy"
)

# Display thermal image
plt.imshow(tir[0], cmap="gray")
plt.title("Thermal Image")
plt.show()

rgb = np.transpose(rgb, (1, 2, 0))

# Normalize RGB values
rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

plt.imshow(rgb)
plt.title("RGB Image")
plt.show()