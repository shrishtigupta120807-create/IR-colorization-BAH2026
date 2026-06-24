import cv2
import argparse
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def load_image(path):
    image = cv2.imread(path)

    if image is None:
        raise ValueError(f"Could not read image: {path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def resize_to_match(predicted, ground_truth):
    if predicted.shape != ground_truth.shape:
        predicted = cv2.resize(
            predicted,
            (ground_truth.shape[1], ground_truth.shape[0])
        )

    return predicted


def calculate_metrics(predicted_path, ground_truth_path):
    predicted = load_image(predicted_path)
    ground_truth = load_image(ground_truth_path)

    predicted = resize_to_match(predicted, ground_truth)

    psnr_value = peak_signal_noise_ratio(
        ground_truth,
        predicted,
        data_range=255
    )

    ssim_value = structural_similarity(
        ground_truth,
        predicted,
        channel_axis=2,
        data_range=255
    )

    return psnr_value, ssim_value


def main():
    parser = argparse.ArgumentParser(
        description="Calculate PSNR and SSIM between predicted RGB and ground truth RGB images."
    )

    parser.add_argument(
        "--predicted",
        required=True,
        help="Path to predicted RGB image"
    )

    parser.add_argument(
        "--ground_truth",
        required=True,
        help="Path to ground truth RGB image"
    )

    args = parser.parse_args()

    psnr_value, ssim_value = calculate_metrics(
        args.predicted,
        args.ground_truth
    )

    print("Evaluation Results")
    print("------------------")
    print(f"PSNR: {psnr_value:.4f}")
    print(f"SSIM: {ssim_value:.4f}")


if __name__ == "__main__":
    main()