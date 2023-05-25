import os
import numpy as np
import cv2
from process import process_video, read_video
from ultralytics import YOLO
from dotenv import load_dotenv


# Load the model
def load_model():
    # load an official detection model
    model = YOLO('yolov8n.pt')
    return model


# Crop video
def crop_video(video_path, filename, directory):

    # Open the video
    cap = cv2.VideoCapture(video_path)

    # Initialise frame counter
    frame_counter = 0

    # Get original video properties, (1280 x 960)
    w_frame, h_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # Define cropping values
    x, y = int(w_frame / 22.5 * 7), int(h_frame / 17 * 9)
    h, w = int(h_frame / 17 * 13), int(w_frame / 22.5 * 10)

    # Rename file
    suffix = "_resized"
    name, ext = os.path.splitext(filename)
    new_filename = 'videos/' + name + suffix + ext
    return_file = directory + '/' + name + suffix + ext

    # Output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(new_filename, fourcc, fps, (w - x, h - y))

    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            # Exit the loop if no frame is read
            break

        # Crop the frame
        cropped_frame = frame[y:h, x:w]

        # Write the cropped frame to the output video
        out.write(cropped_frame)

        # Increment frame counter
        frame_counter += 1

    # Release the video capture and writer objects
    cap.release()
    out.release()

    # Directory to new cropped file
    return return_file


# Read and process the video
def main():
    # Load local .env file for parsing
    load_dotenv()

    # Retrieve file path and filename suffix
    video_path = os.getenv("video_path")
    filename, directory = os.path.basename(
        video_path), os.path.dirname(video_path)

    # Resize and read video into model
    resized_path = crop_video(video_path, filename, directory)
    read_video(resized_path)
    process_video(load_model(), resized_path)


if __name__ == "__main__":
    main()
