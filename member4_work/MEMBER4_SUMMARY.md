# Member 4 Final Summary

## Role

Member 4 was responsible for evaluation, demo support, and final project presentation preparation.

## Work Completed

- Created a separate `member4_work` folder.
- Added evaluation script for PSNR and SSIM.
- Added initial evaluation result file.
- Created a Streamlit demo app.
- Added project dependency list in `requirements.txt`.
- Updated README with instructions to run evaluation and demo.

## Evaluation Work

The evaluation script compares:

- Predicted RGB image
- Ground truth RGB image

Metrics calculated:

- PSNR
- SSIM

Script:

```bash
python member4_work/evaluate_metrics.py --predicted member3_work/prediction.png --ground_truth output/patches/demo/sample_006/rgb_100m_512.png