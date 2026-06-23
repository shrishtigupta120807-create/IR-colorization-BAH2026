from colorization_dataset import ColorizationDataset

dataset = ColorizationDataset()

print("Total Samples:", len(dataset))

tir, rgb = dataset[0]

print("TIR Shape:", tir.shape)

print("RGB Shape:", rgb.shape)