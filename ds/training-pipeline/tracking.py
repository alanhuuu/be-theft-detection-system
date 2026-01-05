import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# load YOLO model
model = YOLO("runs/detect/train2/weights/best.pt")

# initialize DeepSORT
tracker = DeepSort(max_age=30, n_init=3, nn_budget=None, embedder="mobilenet", half=True)

# open your video
cap = cv2.VideoCapture("video-dataset-test/D03F27431DA320251118233323_album_local_cache.MP4")

fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)

frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    # run YOLO detection
    results = model(frame, conf=0.25, imgsz=320)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        conf = float(box.conf)
        w = x2 - x1
        h = y2 - y1
        detections.append(([x1, y1, w, h], conf, "sushi-box"))

    # DeepSORT tracking
    tracks = tracker.update_tracks(detections, frame=frame)

    # draw tracks
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Tracking", frame)

    if cv2.waitKey(delay) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
