# Member 3 Contribution

This folder contains the final colorization work trained on Google Colab.

## Training Details

- Model: U-Net Colorization Network
- Dataset: 305 Thermal-RGB image pairs
- Training Platform: Google Colab
- GPU: T4
- Epochs: 100
- Batch Size: 8
- Loss Function: L1Loss
- Optimizer: Adam
- Best Loss: 0.0738

## Files

- `train_colorization.py` - training script
- `predict_colorization.py` - prediction/inference script
- `colorization_model.py` - U-Net model architecture
- `colorization_model.pth` - trained model weights
- `prediction.png` - sample output image

## Important Note

For final project integration, use the files inside this `member3_work` folder.

The older files in the main project folder are previous versions and should not be treated as the final Member 3 submission.
