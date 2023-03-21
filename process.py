import cv2
from readVid import read_video


def process_video(model, video_path):
    for frame in read_video(video_path):
        results = model(frame)
        persons = []
        for result in results.pred:
            if result[5] == 0:
                label = results.names[result[5]]
                if label == "person":
                    persons.append(result)

        for person in persons:
            x1, y1, x2, y2 = person[:4]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# def process_video(model, video_path, polygon):
#     for frame in read_video(video_path):
#         results = model(frame)

#         persons = []
#         for result in results.pred:
#             if len(result) >= 6 and result[5] == 0:
#                 persons.append(result)

#         # for result in results.pred:
#         #     if result[5] == 0:
#         #         persons.append(result)

#         for person in persons:
#             x1, y1, x2, y2 = person[:4]
#             center_x = int((x1 + x2) / 2)
#             center_y = int((y1 + y2) / 2)

#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             # if cv2.pointPolygonTest(polygon, (center_x, center_y), False) >= 0:
#             #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#         cv2.imshow('Frame', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cv2.destroyAllWindows()
