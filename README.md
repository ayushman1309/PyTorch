# Image Colorization using Convolutional Neural Nets
## Overview
Welcome to the repository, this code provides a solution to a colorization task using Convolutional Neural Networks (CNNs) in PyTorch. This project is aimed at transforming grayscale images to their corresponding colorful images, and it can be used to enhance the quality of old or faded photographs. The code utilizes the latest techniques in deep learning and computer vision to produce visually appealing results. The user can run the code on their own images or use the provided sample dataset to see the results. The project also provides separate training and inference code to control the color temperature of the output images. This allows users to fine-tune the model to their specific needs and preferences. 

## Requirements

- PyTorch (>= 1.7.0)
- Numpy (>= 1.19.3)
- Matplotlib (>= 3.3.3)
- Scikit-Image (>= 0.18.3)
- TorchVision (>= 0.8.1)
- OpenCV-Python (>= 4.5.4)

It is recommended to use a virtual environment for the project to manage the dependencies. The dependencies can be installed by running the following command:
`pip install torch numpy matplotlib scikit-image torchvision opencv-python`

## How to Train the Model

1. Clone this repository using `git clone https://github.com/williamcfrancis/CNN-Image-Colorization-Pytorch.git`

Download the dataset zip file from https://drive.google.com/file/d/15jprd8VTdtIQeEtQj6wbRx6seM8j0Rx5/view?usp=sharing and extract it outside the current directory. The directory structure should look like:

```
│
└───Image_colorization_William
│      train.py
│      basic_model.py
|      colorize_data.py
|      inference_script.py
|      train_hue_control.py
|      colorize_data_hue_control.py
|      Report.pdf
|      README.md  <-- you are here
|
└───train_landscape_images
    │ 
    └─landscape_images
           1.jpg
           2.jpg
```

2. Run train.py with the following arguments:
