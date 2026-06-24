# Member 4 Presentation Notes

## My Role

I handled the evaluation and demo part of the project.

## What I Built

I created:

- An evaluation script for PSNR and SSIM
- A Streamlit demo app
- A requirements file for dependencies
- Documentation for running the demo and evaluation

## Evaluation Explanation

We compare the predicted RGB image with the ground truth RGB image.

The two metrics used are:

### PSNR

PSNR stands for Peak Signal-to-Noise Ratio.

It measures how close the predicted image is to the ground truth image.

Higher PSNR usually means better image quality.

### SSIM

SSIM stands for Structural Similarity Index Measure.

It checks whether the structure, contrast, and texture of the predicted image are similar to the ground truth image.

Higher SSIM means better structural similarity.

## Demo Explanation

The Streamlit demo shows:

- Input infrared image
- Predicted RGB image
- Ground truth RGB image
- Evaluation results

It also includes an upload option to preview a custom TIR image.

## Why This Is Important

The project objective is not only to generate color images, but also to prove that the generated image is useful and close to the real RGB image.

My evaluation and demo help show this clearly.

## Limitations

The current demo uses a fixed sample output for display.

Future work can connect the demo directly to the trained model so uploaded images can be colorized live.