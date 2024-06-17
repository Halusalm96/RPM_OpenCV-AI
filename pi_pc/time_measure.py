import cv2
import socket
import numpy as np
import time

def main():
    # 카메라 설정
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return
    
    # 소켓 설정
    server_address = ('<컴퓨터 IP 주소>', 9999)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        while True:
            start_time = time.time()  # 시작 시간 측정

            # 프레임 읽기
            ret, frame = cap.read()
            if not ret:
                print("프레임을 읽을 수 없습니다.")
                break
            
            # 영상 크기 줄이기
            frame = cv2.resize(frame, (320, 240))
            
            # 흑백으로 변환
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 데이터를 전송하기 위해 직렬화
            _, buffer = cv2.imencode('.jpg', gray_frame)
            data = buffer.tobytes()
            
            # 데이터 전송
            sock.sendto(data, server_address)
            
            end_time = time.time()  # 종료 시간 측정
            elapsed_time = end_time - start_time  # 경과 시간 계산
            print(f"프레임 처리 및 전송 시간: {elapsed_time:.4f} 초")
            
            # 1초 슬립
            time.sleep(1)
    
    finally:
        cap.release()
        sock.close()

if __name__ == "__main__":
    main()
