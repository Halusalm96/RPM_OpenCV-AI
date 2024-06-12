import cv2
from ultralytics import YOLO

def run_webcam():
    # 학습된 모델을 로드합니다
    model = YOLO('/home/user1/Downloads/RPM_OpenCV-AI/yolov8_custom_trained.pt')  # 학습된 모델 파일의 경로를 지정합니다
    
    # 웹캠 연결 (0은 기본 카메라를 의미합니다)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("오류: 웹캠을 열 수 없습니다.")
        return

    while True:
        # 프레임을 하나씩 캡처합니다
        ret, frame = cap.read()
        
        if not ret:
            print("오류: 프레임을 읽을 수 없습니다.")
            break

        # YOLO 모델을 사용하여 프레임을 처리합니다
        results = model(frame)

        # 결과를 프레임에 그립니다
        annotated_frame = results[0].plot()  # plot()은 주석이 달린 프레임을 반환합니다
        
        # 결과 프레임을 표시합니다
        cv2.imshow('YOLOv8 Detection', annotated_frame)
        
        # 'q' 키를 누르면 루프를 종료합니다
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 웹캠을 해제하고 모든 창을 닫습니다
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_webcam()
