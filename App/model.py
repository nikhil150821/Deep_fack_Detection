import torch
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import cv2
from torch import nn
from PIL import Image
import time

# Define your model architecture (example: a simple CNN)
class tModel(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(tModel, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Check the shape of input
        if len(x.shape) == 4:  # If it's a 4D tensor [batch_size, channels, height, width]
            batch_size, c, h, w = x.shape
            seq_length = 1  # Set sequence length to 1 for non-sequential input
        elif len(x.shape) == 5:  # If it's a 5D tensor [batch_size, seq_length, channels, height, width]
            batch_size, seq_length, c, h, w = x.shape
        else:
            raise ValueError(f"Unexpected input tensor shape: {x.shape}")

        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))



# Load your trained model
def load_model(model_path, num_classes, map_location=torch.device('cpu')):
    model = tModel(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=map_location))  # Use map_location here
    model.eval()  # Set the model to evaluation mode
    return model


def predict(model, frames_tensor):
    with torch.no_grad():
        output = model(frames_tensor)  # model's output (fmap, logits)
        logits = output[1]
        _, predicted = torch.max(logits, dim=1)  # Find the predicted class by comparing logits
        
    return ["Fake" if p.item() == 0 else "Real" for p in predicted]


# Prepare image for prediction
def prepare_image(image_path):
    # Load the image using PIL
    image = Image.open(image_path)
    image = image.convert('RGB')  # Ensure RGB format
    image = image.resize((224, 224))  # Resize to match model input size

    # Define the transformation pipeline
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Standard normalization for models like ResNet
    ])

    # Apply transformations and add batch dimension
    image_tensor = transform(image).unsqueeze(0)  # Shape: (1, 3, 224, 224)

    return image_tensor


# Prepare video for prediction (extract frames)
def prepare_video(video_path):
    # Open the video file using OpenCV
    video = cv2.VideoCapture(video_path)
    frames = []

    # Define the transformation pipeline
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # Resize to match model input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Standard normalization
    ])

    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        frame_transformed = transform(frame_rgb)  # Apply transformations
        frames.append(frame_transformed)

    video_tensor = torch.stack(frames)  # Shape: (num_frames, 3, 224, 224)
    video_tensor = video_tensor.unsqueeze(0)  # Add batch dimension (1, num_frames, 3, 224, 224)

    return video_tensor


# Function to get predictions for images
def predict_image(model, image_path):
    image = prepare_image(image_path)
    with torch.no_grad():
        output = model(image)  # output is a tuple (fmap, logits)
        logits = output[1]  # Access the logits (second element of the tuple)
        _, predicted = torch.max(logits, dim=1)  # Get the predicted class
        
    return "Fake" if predicted.item() == 0 else "Real"



# Function to get predictions for videos
def predict_video(model, video_path):
    # Initialize face detection (Haar Cascade or other method)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Read the video
    video_capture = cv2.VideoCapture(video_path)
    
    frames_with_faces = 0
    total_frames = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            frames_with_faces += 1
        
        total_frames += 1

    video_capture.release()

    faces_detected = frames_with_faces > 0
    output = "REAL" if faces_detected else "FAKE"  # Adjust output based on face detection

    return faces_detected, output # Returns a list of predictions for each frame


def validation_dataset(video_paths):
    frames = []
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Ensure the frame is a PIL Image before transformations
        transforms.Resize((128, 128)),  # Resize to match model input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            frame_transformed = transform(frame_rgb)  # Apply transformations
            frames.append(frame_transformed)  # Add transformed frame to list
        cap.release()
    
    # Stack frames into a tensor with shape [batch_size, channels, height, width]
    frames_tensor = torch.stack(frames)
    frames_tensor = frames_tensor.unsqueeze(0)  # Add batch dimension if necessary
    return frames_tensor



# Example usage
if __name__ == "__main__":
    model_path = 'model_97_acc_80_frames_FF_data.pt'
    model = load_model(model_path, num_classes=2)  # Adjust num_classes as needed

    # Example for an image
    image_path = 'path_to_your_image.jpg'  # Replace with your image path
    print("Image Prediction:", predict_image(model, image_path))

    # Example for a video
    video_path = 'path_to_your_video.mp4'  # Replace with your video path
    print("Video Predictions:", predict_video(model, video_path))
