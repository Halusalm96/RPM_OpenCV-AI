# 필요한 라이브러리 임포트
from ultralytics import YOLO

# 사전 학습된 YOLO v8 모델 로드
model = YOLO('yolov8n.pt')  # 'yolov8n.pt'는 사전 학습된 모델 파일 경로입니다.


data_yaml_content = """
train: /home/user1/Downloads/RPM_OpenCV-AI/play/train
val: /home/user1/Downloads/RPM_OpenCV-AI/play/val
nc: 4
names: [viking, 'roller', 'wheel', 'round']
"""

# data.yaml 파일 저장
with open('coco8-seg.yaml', 'w') as f:
    f.write(data_yaml_content)

# 모델 파인튜닝 (전이 학습)
model.train(data='coco8-seg.yaml', epochs=30, batch=16, imgsz=640)

# 모델 저장
model.save('yolov8_custom_trained.pt')
