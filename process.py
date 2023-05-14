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
    # Define and initiate the polygon coordinates
    # polygon = np.array([
    #     [409, 590], 
    #     [417, 676], 
    #     [468, 669], 
    #     [463, 585]
    # ])

    polygon = np.array([
        [350, 500], 
        [350, 750], 
        [500, 750], 
        [500, 500]
    ])

    video_info = sv.VideoInfo.from_video_path(video_path=video_path)
    zone = sv.PolygonZone(polygon = polygon, frame_resolution_wh = video_info.resolution_wh)

    # Creates an instance of the BoxAnnotator class from supervision to detect and label
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=2
    )
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone, 
        color=sv.Color.blue(), 
        thickness=2,
        text_thickness=2,
        text_scale=2
    )

    for frame in read_video(video_path):
        results = model(frame)[0]
        detections = sv.Detections.from_yolov8(results)
        # detections = detections.filter_by_polygon(polygon)
        detections = detections[detections.class_id == 0]
        zone.trigger(detections=detections)
        
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