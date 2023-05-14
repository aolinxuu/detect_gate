from process import process_video, read_video
from ultralytics import YOLO


# Load the model
def load_model():
    model = YOLO('yolov8n.pt')  # load an official detection model
    return model

video_path = "C:\\Users\\aolin\\Videos\\sample1.mp4"
read_video(video_path)
process_video(load_model(), video_path)
