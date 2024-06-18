import cv2
import socket
import numpy as np
import pymysql
from ultralytics import YOLO

# YOLO 모델 로드
model = YOLO('yolov8n.pt')
model.to('cpu')

# 데이터베이스 연결 설정
try:
    db = pymysql.connect(
        host="13.124.83.151",
        user="root",
        password="1235",
        database="rpm"
    )
except pymysql.MySQLError as e:
    print(f"Error connecting to the database: {e}")
    db = None

def save_to_database(detected_objects):
    if db is None:
        print("Database connection is not available.")
        return

    cursor = db.cursor()
    
    for obj in detected_objects:
        class_name = obj['name']
        confidence = obj['confidence']

        # SQL 쿼리 작성 및 실행
        sql = """
        INSERT INTO detections (detections_key, class_name, confidence)
        VALUES (%s, %s, %s)
        ON DUPLICATE KEY UPDATE class_name = VALUES(class_name), confidence = VALUES(confidence)
        """
        cursor.execute(sql, ('1', class_name, confidence))  # '1' 키를 사용하여 하나의 행만 삽입 또는 업데이트

    db.commit()
    cursor.close()

def main():
    # 소켓 설정
    server_address = ('', 9999)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(server_address)
    
    print("데이터를 기다리는 중...")
    
    try:
        while True:
            # 데이터 수신
            data, address = sock.recvfrom(65535)
            
            # 수신된 데이터를 디코딩하여 이미지로 변환
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # gray로 하면 yolo에서 형식이 맞지 않다는 오류 생성
            
            # YOLO v8을 이용한 객체 탐지
            results = model(frame)
            
            detected_objects = []
            for result in results:
                for box in result.boxes:
                    confidence = float(box.conf)
                    if confidence >= 0.5:  # 정확도가 50% 이상인 경우에만 추가
                        detected_objects.append({
                            'name': result.names[int(box.cls)],
                            'confidence': confidence
                            # 'box': [int(box.xyxy[0]), int(box.xyxy[1]), int(box.xyxy[2]-box.xyxy[0]), int(box.xyxy[3]-box.xyxy[1])]
                        })
            
            # 데이터베이스에 저장
            save_to_database(detected_objects)
    
    finally:
        sock.close()
        if db is not None:
            db.close()

if __name__ == "__main__":
    main()
