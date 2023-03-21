import cv2


def read_video(video_path):
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        print('Error opening video file')
        exit()

    while True:
        ret, frame = vid.read()
        if not ret:
            break

        yield frame

    vid.release()
