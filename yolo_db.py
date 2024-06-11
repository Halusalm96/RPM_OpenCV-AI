from ultralytics import YOLO
from PIL import Image as PILImage
import mysql.connector
import cv2

# YOLOv8 모델 로드
model = YOLO('yolov8s.pt')

# MySQL 연결 설정
db_conn = mysql.connector.connect(
    host="localhost",
    user="new_user",  # 새로 생성한 MySQL 사용자 이름으로 변경하세요
    password="new_password",  # 새로 생성한 MySQL 비밀번호로 변경하세요
    database="yolo_db"
)
cursor = db_conn.cursor()

def save_detection_to_db(detections):
    for detection in detections:
        sql = "INSERT INTO detections (class_name, confidence, xmin, ymin, xmax, ymax) VALUES (%s, %s, %s, %s, %s, %s)"
        values = (
            detection['class'],
            detection['confidence'],
            detection['xmin'],
            detection['ymin'],
            detection['xmax'],
            detection['ymax']
        )
        cursor.execute(sql, values)
        db_conn.commit()

def detect_and_save(frame):
    # OpenCV 이미지를 PIL 이미지로 변환
    pil_img = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # YOLOv8 모델로 객체 탐지 수행
    results = model(pil_img)

    # 결과를 처리하고 필요한 정보를 추출
    detections = []
    for result in results:
        for box in result.boxes:
            detection = {
                'class': result.names[int(box.cls)],
                'confidence': float(box.conf),
                'xmin': float(box.xyxy[0][0]),
                'ymin': float(box.xyxy[0][1]),
                'xmax': float(box.xyxy[0][2]),
                'ymax': float(box.xyxy[0][3]),
            }
            detections.append(detection)
    
    # 데이터베이스에 탐지 결과 저장
    save_detection_to_db(detections)

if __name__ == "__main__":
    # 카메라 캡처 객체 생성
    cap = cv2.VideoCapture(0)  # 0은 기본 카메라를 의미합니다.

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 객체 탐지 및 결과 저장
        detect_and_save(frame)

        # 화면에 결과 표시 (선택 사항)
        cv2.imshow('YOLOv8 Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()