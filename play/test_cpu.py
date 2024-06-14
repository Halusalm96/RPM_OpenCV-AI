import cv2
from ultralytics import YOLO

def run_webcam():
    try:
        # 학습된 모델을 로드합니다 (CPU 모드로)
        model = YOLO('/home/user1/Downloads/RPM_OpenCV-AI/yolov8_custom_trained.pt')
        model.to('cpu')  # 모델을 CPU로 이동
        print("모델이 성공적으로 로드되었습니다.")
    except Exception as e:
        print(f"모델 로드 중 오류가 발생했습니다: {e}")
        return
    
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

        try:
            # YOLO 모델을 사용하여 프레임을 처리합니다 (CPU 모드)
            results = model(frame)
            
            # 결과를 프레임에 그립니다
            annotated_frame = results[0].plot()  # plot()은 주석이 달린 프레임을 반환합니다
        except Exception as e:
            print(f"프레임 처리 중 오류가 발생했습니다: {e}")
            break
        
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
