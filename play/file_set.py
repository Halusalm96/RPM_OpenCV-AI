from ultralytics import YOLO
# import matplotlib.pyplot as plt
# import cv2
import torch

# YOLOv8 모델 로드
model = YOLO('yolov8n-seg.pt')

# CUDA 지원 여부 확인
if torch.cuda.is_available():
    # GPU 사용 설정
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# 모델을 GPU로 이동
model.to(device)

model.train(data='/home/user1/Downloads/RPM_OpenCV-AI/play/PLAY.yaml',epochs=15, patience=10, batch=32, imgsz=416)

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