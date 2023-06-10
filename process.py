import cv2
import numpy as np

def read_video(video_path):
    vid = cv2.VideoCapture(video_path)
    print('successfully opened video')
    if not vid.isOpened():
        print('Error opening video file')
        exit()

    count = 0
    while True:
        ret, frame = vid.read()
        if not ret:
            break
        count += 1
        if count % 3 != 0:
            continue
        yield frame

    vid.release()


def process_video(model, video_path):
    # initialize HOG descriptor 
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    for frame in read_video(video_path):
        # frame = cv2.resize(frame, (640, 480))



        # detect humans in input image
        humans, _ = hog.detectMultiScale(frame, winStride=(10, 10),
                                          padding=(32, 32), scale=1.1)

        # getting no. of human detected
        print('Human Detected:', len(humans))

        # loop over all detected humans
        for (x, y, w, h) in humans:
            pad_w = int(0.15 * w)
            pad_h = int(0.01 * h)
            cv2.rectangle(frame, (x + pad_w, y + pad_h), (x + w - pad_w, y + h - pad_h), (0, 255, 0), 2)
            # Draw label
            cv2.putText(frame, 'Human', (x + pad_w, y + pad_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(30) == 27:
            break

    cv2.destroyAllWindows()