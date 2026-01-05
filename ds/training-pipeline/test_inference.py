from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train2/weights/best.pt")

img = "test-dataset/WYZE_1763525189.179735.JPG"
results = model(img, save=True)

print(results)
