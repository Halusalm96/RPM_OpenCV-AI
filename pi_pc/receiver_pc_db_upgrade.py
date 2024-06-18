import cv2
import socket
import numpy as np
import pymysql
from ultralytics import YOLO

# YOLO 모델 로드
model = YOLO('yolov8n.pt')
model.to('cpu')

# 데이터베이스 연결 설정
db = pymysql.connect(
    host="localhost",
    user="new_user",  
    password="new_password",  
    database="yolo_db2"
)

def save_to_database(detected_objects):
    cursor = db.cursor()
    for obj in detected_objects:
        class_name = obj['name']
        confidence = obj['confidence']
        # x, y, w, h = obj['box']

        # SQL 쿼리 작성 및 실행
        # sql = "INSERT INTO detections (class_name, confidence, x, y, width, height) VALUES (%s, %s, %s, %s, %s, %s)"
        # cursor.execute(sql, (class_name, confidence, x, y, w, h))
        sql = "INSERT INTO detections (class_name, confidence) VALUES (%s, %s)"
        cursor.execute(sql, (class_name, confidence))

    db.commit()

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
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)   #gray로 하면 yolo에서 형식이 맞지 않다는 오류 생성
            
            # YOLO v8을 이용한 객체 탐지
            results = model(frame)
            
            detected_objects = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    detected_objects.append({
                        'name': result.names[int(box.cls)],
                        'confidence': float(box.conf)   #,
                        # 'box': [int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]-box.xyxy[0][0]), int(box.xyxy[0][3]-box.xyxy[0][1])]
                    })
            
            # 데이터베이스에 저장
            save_to_database(detected_objects)
            
            # # 탐지 결과 표시 (옵션)
            # for obj in detected_objects:
            #     x, y, w, h = obj['box']
            #     cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            #     cv2.putText(frame, f"{obj['name']} {obj['confidence']:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # 영상 표시
            cv2.imshow('YOLO Detection', frame)
            if cv2.waitKey(1) == ord('q'):
                break
    
    finally:
        sock.close()
        db.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
