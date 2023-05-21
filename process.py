import cv2
import supervision as sv
import numpy as np


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

    polygon = np.array([
        [410, 675],
        [463, 665]
    ])


    video_info = sv.VideoInfo.from_video_path(video_path=video_path)
    zone = sv.PolygonZone(polygon = polygon, frame_resolution_wh = video_info.resolution_wh)

    # Creates an instance of the BoxAnnotator class from supervision to detect and label
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=1
    )
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone, 
        color=sv.Color.blue(), 
        thickness=2,
        text_thickness=1,
        text_scale=0.5
    )

    for frame in read_video(video_path):
        results = model(frame)[0]
        detections = sv.Detections.from_yolov8(results)
        # detections = detections.filter_by_polygon(polygon)
        detections = detections[detections.class_id == 0]
        zone.trigger(detections=detections)

        box_coordinates = detections.xyxy

        for box in box_coordinates:
            x1, y1, x2, y2 = box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            center_coordinate = (center_x, center_y)

            # Add dot at center coordinate
            cv2.circle(frame, center_coordinate, 3, (0, 0, 255), -1)

        box_labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _ in detections
        ]

        frame = box_annotator.annotate(
            scene=frame, detections=detections, labels=box_labels)
        
        frame = zone_annotator.annotate(frame)

        cv2.imshow('Frame', frame)
        if (cv2.waitKey(30) == 27):
            break

    cv2.destroyAllWindows()