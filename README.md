# ASL Alphabet Image Classification

## 📌 Description

This project implements an image classification model for American Sign Language (ASL) alphabets using the TinyVGG architecture. The model is trained on the ASL Alphabet dataset and implemented in PyTorch.

## 📂 Dataset

The dataset used for training and testing is available on Kaggle:\
[ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

## 🏗️ Model Architecture

The model is based on the TinyVGG architecture. Learn more about it here:\
[TinyVGG Architecture](https://poloclub.github.io/cnn-explainer/)

## 🎯 Model Performance

- Achieved **100% accuracy** on the test dataset from Kaggle.
- In real-time webcam detection, the model takes approximately **2 seconds per classification** due to processing time.

## 📦 Requirements

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install torch torchvision torchinfo numpy pandas matplotlib tqdm Pillow pathlib scikit-learn opencv-python cvzone
```

## 🚀 Real-Time ASL Detection

The real-time ASL detection is implemented using OpenCV for webcam capture and the **cvzone** library for hand tracking. The system detects a hand, crops the region of interest, preprocesses it, and feeds it into the trained TinyVGG model for classification. The detected sign is then displayed on the webcam feed. 

While the model performs exceptionally well on the test dataset, real-time classification takes around **2 seconds per frame**, likely due to preprocessing, model inference time, and system limitations.

## 🔥 Future Improvements

- Optimize real-time detection speed to improve responsiveness.
- Deploy the model as a lightweight web or mobile application.

## ⚡ GPU Used

- CUDA Version: **2.6.0+cu126**
- CUDA Download Link: [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads)

---

Feel free to contribute or report issues! 🚀

