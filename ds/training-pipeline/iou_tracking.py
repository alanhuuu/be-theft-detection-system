import cv2
import time
from ultralytics import YOLO

#############################################
# IOU-Based Tracker (lightweight & fast)
#############################################

class IOUTracker:
    def __init__(self, iou_threshold=0.3, max_age=12):
        self.tracks = {}         # track_id → bbox
        self.ages = {}           # track_id → frames since last seen
        self.next_id = 0
        self.iou_threshold = iou_threshold
        self.max_age = max_age

    def iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH

        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        unionArea = areaA + areaB - interArea + 1e-6
        return interArea / unionArea

    def update(self, detections):
        new_tracks = {}
        used_ids = set()

        for det in detections:
            best_iou = 0
            best_id = None

            for tid, tbox in self.tracks.items():
                iou_score = self.iou(det, tbox)
                if iou_score > best_iou:
                    best_iou = iou_score
                    best_id = tid

            if best_iou > self.iou_threshold:
                new_tracks[best_id] = det
                used_ids.add(best_id)
            else:
                new_tracks[self.next_id] = det
                self.next_id += 1

        # age tracks
        for tid in list(self.tracks.keys()):
            if tid not in new_tracks:
                self.ages[tid] = self.ages.get(tid, 0) + 1
                if self.ages[tid] < self.max_age:
                    new_tracks[tid] = self.tracks[tid]
                else:
                    del self.ages[tid]

        self.tracks = new_tracks
        return new_tracks


#############################################
# YOLO + IOU Tracking Pipeline
#############################################

model = YOLO("runs/detect/train2/weights/best.pt")

tracker = IOUTracker(iou_threshold=0.3, max_age=12)

cap = cv2.VideoCapture("video-dataset-test/D03F27431DA320251118233323_album_local_cache.MP4")
fps_video = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps_video)

prev = time.time()

frame_i = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_i += 1

    # YOLO inference (small imgsz = FAST)
    results = model(frame, conf=0.25, imgsz=320)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append([x1, y1, x2, y2])

    # IOU tracking
    tracks = tracker.update(detections)

    # Draw tracks
    for tid, bbox in tracks.items():
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
        cv2.putText(frame, f"ID {tid}", (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # Show FPS
    now = time.time()
    fps = 1 / (now - prev)
    prev = now
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    cv2.imshow("IOU Tracking", frame)

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
