import os
import cv2
# from process import process_video, read_video
from process import Tracker
from ultralytics import YOLO
from dotenv import load_dotenv


# Load the model
def load_model():
    model = YOLO('yolov8n.pt')
    return model


# Read and process the video
def main():
    load_dotenv()
    video_path = os.getenv("video_path")
    output_directory = os.path.dirname(video_path)  # Get the directory of the input video

    # Create a copy of the video in the same folder
    video_filename = os.path.basename(video_path)
    video_copy_path = os.path.join(output_directory, "copy_" + video_filename)

    # Open the original video
    cap = cv2.VideoCapture(video_path)

    # Get the video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )

    # Define the cropping values
    x, y = int(frame_size[0] / 22.5 * 7), int(frame_size[1] / 17 * 9)
    h, w = int(frame_size[1] / 17 * 13), int(frame_size[0] / 22.5 * 10)

    # Create a video writer to save the cropped copy
    cropped_frame_size = (w - x, h - y)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_copy_path, fourcc, fps, cropped_frame_size)

    # Read, crop, and save each frame of the video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cropped_frame = frame[y:h, x:w]  # Crop the frame
        out.write(cropped_frame)  # Save the cropped frame

    # Release the video capture and writer objects
    cap.release()
    out.release()
    tracker_instance = Tracker()
    # Read and process the cropped video
    tracker_instance.read_video(video_copy_path)
    tracker_instance.process_video(load_model(), video_copy_path)


if __name__ == "__main__":
    main()
