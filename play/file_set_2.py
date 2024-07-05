from ultralytics import YOLO
# import matplotlib.pyplot as plt
# import cv2
# import torch

# YOLOv8 모델 로드
model = YOLO('yolov8n.pt')

model.train(data='/Users/503/play/coco8.yaml',epochs=50, patience=10, batch=32, imgsz=416)

# 데이터셋 로드
# dataset = model.load_data(data='/Users/503/play/PLAY.yaml',epochs=15, patience=10, batch=32, imgsz=416)

# 모델 학습
# model.train(data=dataset, epochs=20)

# # 데이터셋의 첫 번째 샘플 확인
# for img, label in dataset:
#     img = cv2.cvtColor(img.numpy(), cv2.COLOR_BGR2RGB)
#     plt.imshow(img)
#     plt.show()
#     print(label)
#     break

# 학습된 클래스 이름 출력
print(type(model.names), len(model.names))
print(model.names)