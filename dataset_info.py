from pathlib import Path

dataset_path = Path("output/patches")

sample_count = 0

for scene in dataset_path.iterdir():
    if scene.is_dir():
        for sample in scene.iterdir():
            if sample.is_dir():
                sample_count += 1

print("Total samples:", sample_count)