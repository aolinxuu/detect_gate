import os
import cv2
from process import process_video, read_video
from ultralytics import YOLO
from dotenv import load_dotenv


def load_model():
    """Load the YOLO model."""
    return YOLO('yolov8n.pt')


def get_video_properties(video_path):
    """Get properties of the video."""
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    return cap, fps, frame_size


def crop_video(cap, fps, frame_size, video_copy_path):
    """Crop the video and save to a new file."""
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


def process_cropped_video(video_copy_path):
    """Read and process the cropped video."""
    read_video(video_copy_path)
    process_video(load_model(), video_copy_path)


def main():
    """Main function."""
    load_dotenv()
    video_path = os.getenv("video_path")
    # Get the directory of the input video
    output_directory = os.path.dirname(video_path)

    # Create a copy of the video in the same folder
    video_filename = os.path.basename(video_path)
    video_copy_path = os.path.join(output_directory, "copy_" + video_filename)

    cap, fps, frame_size = get_video_properties(video_path)
    crop_video(cap, fps, frame_size, video_copy_path)
    process_cropped_video(video_copy_path)


if __name__ == "__main__":
    main()
