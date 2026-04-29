import argparse
import torch
import matplotlib.pyplot as plt
from skimage.color import lab2rgb, rgb2gray
import numpy as np
import os
import cv2

# Device setup (CPU/GPU safe)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ✅ Function was missing in your code
def to_rgb(grayscale_input, ab_input):
    plt.clf()
    color_image = torch.cat((grayscale_input, ab_input), 0).numpy()
    color_image = color_image.transpose((1, 2, 0))

    # Convert LAB → RGB
    color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
    color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128
    color_image = lab2rgb(color_image.astype(np.float64))

    plt.imsave(arr=color_image, fname='inference/inference_output.jpg')


if __name__ == '__main__':
    os.makedirs('inference/', exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='models/saved_model.pth', type=str)
    parser.add_argument('--image_path', default='inference/0.jpg', type=str)

    args = parser.parse_args()

    print('Beginning Inference')

    # Load model
    model = torch.load(args.model_path, weights_only=False)
    model = model.to(device)
    model.eval()

    # Load image
    input_img = cv2.imread(args.image_path)

    if input_img is None:
        raise Exception("❌ Image not found. Check path!")

    input_img = cv2.resize(input_img, (256, 256))
    input_gray = rgb2gray(input_img)

    # Convert to tensor
    input_gray = torch.from_numpy(input_gray).unsqueeze(0).float()
    input_gray = torch.unsqueeze(input_gray, dim=0).to(device)

    # Prediction
    with torch.no_grad():
        output_ab = model(input_gray)

    # Convert to RGB
    to_rgb(input_gray[0].cpu(), output_ab[0].cpu())

    print("✅ Colorized image saved at 'inference/inference_output.jpg'")