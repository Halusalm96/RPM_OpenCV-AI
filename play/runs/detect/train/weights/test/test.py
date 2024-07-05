from ultralytics import YOLO
import torch

# YOLOv8 모델 로드 (사전 학습된 모델 사용)
model = YOLO('model.pt')

# 모델 학습 (생략 가능, 이미 학습된 모델을 사용하는 경우)
# model.train(data='data.yaml', epochs=50, batch_size=16, imgsz=640)

# 모델을 평가 모드로 설정 (학습된 모델을 사용할 때)
model.model.eval()

# 더미 입력 생성
dummy_input = torch.randn(1, 3, 640, 640)

# ONNX로 모델 내보내기
torch.onnx.export(model.model, dummy_input, 'model.onnx', opset_version=12)
