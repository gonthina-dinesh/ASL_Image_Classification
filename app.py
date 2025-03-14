import cv2
import torch
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import torchvision.transforms as transforms
from PIL import Image
from torch import nn

# Load the trained PyTorch model
class TinyVGG(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*32*32, out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))

# Load trained model
model = torch.jit.load("asl_model.pt", map_location=torch.device('cpu'))
model.eval()

# Load class labels
labels = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "NOTHING", "SPACE"
]

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize hand detector
detector = HandDetector(maxHands=1)

# Image processing parameters
offset = 40  # Padding around detected hand
crop_size = 200  # Fixed crop size for consistency
imgSize = 128  # Resize for model input

# Define preprocessing transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((imgSize, imgSize)),  # Resize to model's input size
    transforms.ToTensor(),
])

while True:
    success, img = cap.read()
    if not success:
        continue
    
    imgOutput = img.copy()
    hands = detector.findHands(img, draw=False)[0]  # Detect hand

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        
        # Ensure cropping remains within image boundaries
        y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
        x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
        imgCrop = img[y1:y2, x1:x2]
        
        if imgCrop.size == 0:
            continue
        
        # Resize to 200x200 before further processing
        imgCrop = cv2.resize(imgCrop, (crop_size, crop_size))
        
        # Convert to PyTorch tensor
        imgTensor = transform(imgCrop).unsqueeze(0)  # Add batch dimension

        # Predict using the model
        with torch.no_grad():
            output = model(imgTensor)
            predicted_index = torch.argmax(output).item()

        # Display prediction
        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                      (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[predicted_index], (x, y - 46), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)
        
        cv2.imshow("ImageCrop", imgCrop)
    
    cv2.imshow("Image", imgOutput)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()