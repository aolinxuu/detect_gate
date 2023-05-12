import cv2
import supervision as sv

def read_video(video_path):
    vid = cv2.VideoCapture(video_path)
    print('sucessfully opened video')
    if not vid.isOpened():
        print('Error opening video file')
        exit()

    while True:
        ret, frame = vid.read()
        if not ret:
            break

        yield frame

    vid.release()

def process_video(model, video_path):
    # Creates an instance of the BoxAnnotator class from supervision to detect and label
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=2
    )

    for frame in read_video(video_path):
        results = model(frame)[0]
        detections = sv.Detections.from_yolov8(results)
        
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _ in detections
        ]

        frame = box_annotator.annotate(
            scene=frame, detections=detections, labels=labels)

        cv2.imshow('Frame', frame)
        if (cv2.waitKey(30) == 27):
            break

    cv2.destroyAllWindows()