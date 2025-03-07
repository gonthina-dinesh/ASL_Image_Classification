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

- Achieved **98.5% accuracy** on the test dataset.

## 📦 Requirements

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install torch torchvision numpy pandas matplotlib pathlib pillow torchinfo tqdm
```

## 🚀 How to Run

### 1️⃣ Clone the Repository

```bash
git clone <repo-link>
cd <repo-folder>
```

### 2️⃣ Train the Model

Run the Jupyter Notebook to train the model:

```bash
jupyter notebook
```

Open and execute the training notebook.

### 3️⃣ Inference

Load the trained model (`.pth` file) and run inference on new images.

```python
import torch
from PIL import Image
from torchvision import transforms

# Load model
model = torch.load("model.pth")
model.eval()

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
img = Image.open("path/to/image.jpg")
img = transform(img).unsqueeze(0)

# Predict
output = model(img)
prediction = torch.argmax(output, dim=1)
classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "nothing", "space"]
print(f"Predicted Class: {classes[prediction.item()]}")
```

## 🔥 Future Improvements

- Implement real-time sign language detection using a webcam.
- Optimize model for better performance and efficiency.

## ⚡ GPU Used

- CUDA Version: **2.6.0+cu126**
- CUDA Download Link: [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads)

---

Feel free to contribute or report issues! 🚀

