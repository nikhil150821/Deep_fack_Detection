import cv2
import torch
import os
from torchvision import transforms

# Configuration for frame extraction and transformation
MAX_FRAMES = 80  # Example max number of frames to process
IMAGE_SIZE = (224, 224)  # Example input size for model

# Transformation pipeline (adjust based on your model's training setup)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])

def preprocess_video(video_path):
    """Extract frames from video, detect faces, and prepare tensor input."""
    frames = []
    preprocessed_images = []
    faces_cropped_images = []
    no_faces = False

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_extract = min(frame_count, MAX_FRAMES)
    frame_interval = frame_count // frames_to_extract  # Skip frames to reach max frames

    while len(frames) < frames_to_extract:
        ret, frame = cap.read()
        if not ret:
            break

        # Process every 'frame_interval' frames
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % frame_interval == 0:
            # Detect faces in frame
            faces = detect_faces(frame)
            if faces:
                for face in faces:
                    face_img = frame[face[1]:face[3], face[0]:face[2]]
                    face_img_resized = cv2.resize(face_img, IMAGE_SIZE)
                    faces_cropped_images.append(face_img_resized)
                    frames.append(transform(face_img_resized))
                    preprocessed_images.append(frame)
            else:
                no_faces = True  # Flag if no faces found in any frame

    cap.release()

    # Convert frames list to tensor
    if frames:
        frames_tensor = torch.stack(frames).unsqueeze(0)  # Batch dimension
    else:
        frames_tensor = torch.empty(0)  # Empty tensor if no frames

    return frames_tensor, preprocessed_images, faces_cropped_images, no_faces

def detect_faces(frame):
    """Detect faces in a given frame using OpenCV's pre-trained Haar cascades."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Convert detected faces to [x1, y1, x2, y2] format for cropping
    face_coords = [(x, y, x + w, y + h) for (x, y, w, h) in faces]
    return face_coords
