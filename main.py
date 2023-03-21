from readVid import read_video
from process import process_video
import numpy as np
import torch
from dotenv import load_dotenv
import os

# Define the polygon coordinates
polygon = np.array([
    [320, 270], [328, 346], [372, 338], [364, 262], [320, 270]
])

# Load the model


def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model


# Read and process the video
load_dotenv()
video_path = os.getenv('video')
read_video(video_path)
process_video(load_model(), video_path, polygon)
