from ultralytics import YOLO
from PIL import Image as PILImage
import mysql.connector
import cv2

# YOLOv8 모델 로드
model = YOLO('yolov8s.pt')
model.to('cpu')

# MySQL 연결 설정
db_conn = mysql.connector.connect(
    host="localhost",   # SQL url 입력시 외부 DB사용 가능
    user="new_user",  # 새로 생성한 MySQL 사용자 이름으로 변경하세요
    password="new_password",  # 새로 생성한 MySQL 비밀번호로 변경하세요
    database="yolo_db2"
)
cursor = db_conn.cursor()

def save_detection_to_db(detections):
    for detection in detections:
        sql = "INSERT INTO detections (class_name, confidence) VALUES (%s, %s)"
        values = (
            detection['class'],
            detection['confidence'],
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
        for i in range(1, 100):
            cv2.imshow('YOLOv8 Detection', frame)
        

        if cv2.waitKey(350) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()