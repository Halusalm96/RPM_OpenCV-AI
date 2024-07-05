from ultralytics import YOLO
# import matplotlib.pyplot as plt
# import cv2
import torch

# YOLOv8 모델 로드
model = YOLO('yolov8n.pt')

model.train(data='/Users/503/play/PLAY.yaml',epochs=15, patience=10, batch=32, imgsz=416)

# 학습된 클래스 이름 출력
print(type(model.names), len(model.names))
print(model.names)