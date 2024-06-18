import time
from ultralytics import YOLO
from PIL import Image as PILImage
import mysql.connector
import cv2
import torch

# YOLOv8 모델 로드
model = YOLO('yolov8s.pt')
device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# 외부 MySQL 연결 설정
db_conn = mysql.connector.connect(
    host="localhost",  # 외부 데이터베이스 호스트 주소
    user="new_user",  # 외부 MySQL 사용자 이름
    password="new_password",  # 외부 MySQL 비밀번호
    database="yolo_db2"#,
    # ssl_ca='path_to_ca_cert',  # CA 인증서 경로 (선택 사항)
    # ssl_cert='path_to_client_cert',  # 클라이언트 인증서 경로 (선택 사항)
    # ssl_key='path_to_client_key'  # 클라이언트 키 경로 (선택 사항)
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
    # pil_img = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # 영상 크기 줄이기
    frame = cv2.resize(frame, (320, 240))
            
    # 흑백으로 변환
    gray_frame = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    # YOLOv8 모델로 객체 탐지 수행
    start_inference_time = time.time()

    results = model(gray_frame)
    # results = model(pil_img)
    end_inference_time = time.time()

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

    return end_inference_time - start_inference_time

if __name__ == "__main__":
    # 카메라 캡처 객체 생성
    cap = cv2.VideoCapture(0)  # 0은 기본 카메라를 의미합니다.

    frame_skip = 0  # 처리할 프레임 건너뛰기 설정
    frame_count = 0
    wait_key_time = 1000  # 기본 waitKey 시간 (ms)

    while True:
        loop_start_time = time.time()  # 루프 시작 시간 측정

        ret, frame = cap.read()
        if not ret:
            break

        # 지정한 프레임 간격마다 처리
        # if frame_count % frame_skip == 0:
        #     inference_time = detect_and_save(frame)
        inference_time = detect_and_save(frame)

        frame_count += 1

        # key = cv2.waitKey(wait_key_time) & 0xFF
        loop_end_time = time.time()  # 루프 종료 시간 측정

        # if key == ord('q'):
        #     break

        loop_time = loop_end_time - loop_start_time
        total_time = loop_time + (wait_key_time / 1000.0)
        print(f"Loop Time: {loop_time:.2f} seconds, Total Estimated Time: {total_time:.2f} seconds")
        # total_time = loop_time + inference_time + (wait_key_time / 1000.0)
        # print(f"Loop Time: {loop_time:.2f} seconds, Inference Time: {inference_time:.2f} seconds, Total Estimated Time: {total_time:.2f} seconds")

    cap.release()
    cv2.destroyAllWindows()

    # MySQL 연결 종료
    cursor.close()
    db_conn.close()