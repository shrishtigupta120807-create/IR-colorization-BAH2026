from pathlib import Path
import numpy as np
from torch.utils.data import Dataset


class ColorizationDataset(Dataset):

    def __init__(self, root_dir="output/patches"):

        self.samples = []

        root = Path(root_dir)

        for scene in root.iterdir():

            if scene.is_dir():

                for sample in scene.iterdir():

                    if sample.is_dir():

                        tir_file = sample / "tir_100m_512.npy"
                        rgb_file = sample / "rgb_100m_512.npy"

                        self.samples.append(
                            (tir_file, rgb_file)
                        )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        tir_path, rgb_path = self.samples[idx]

        tir = np.load(tir_path).astype(np.float32)

        rgb = np.load(rgb_path).astype(np.float32)

        return tir, rgb