import os
from process import process_video, read_video
import numpy as np
from ultralytics import YOLO
from dotenv import load_dotenv


# Define the polygon coordinates
polygon = np.array([
    [320, 270], [328, 346], [372, 338], [364, 262], [320, 270]
])

# Load the model
def load_model():
    model = YOLO('yolov8n.pt')  # load an official detection model
    return model


# Read and process the video
load_dotenv()
video_path = os.getenv("video_path")

read_video(video_path)
process_video(load_model(), video_path)
