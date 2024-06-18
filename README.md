의존성 : OpenCV, YOLO v8(python3.8)

2024.6.11 16:00 jinwoong yolo_db.py로 DB에 데이터 보냄
2024.6.12 10:13 jinwoong yolo_db_upgrade.py로 검출 간격을 늘려서 영상 정보 전달 속도 향상 + 불필요한 좌표값 삭제

2024.6.11 16:00 jinwoong
yolo_db.py로 DB에 데이터 보냄
 766520d065d1f8b0a722e031dc3316badfa1b802

2024.6.14 10:50 jinwoong yolo_db_upgrade2.py로 영상 측정 시간을 늘려 부하를 줄임
2024.6.14 10:50 jinwoong yolo_db_upgrade2_time_check.py를 통해 함수 호출 시간과 대기시간을 통해 대략적인 프로그램 순환 시간 파악 가능
 HEAD
                         만약 이 걸로도 부하가 크면 cv2.resize를 통해 이미지 사이즈 줄이는 것 필요인
2024.6.14 12:20 jinwoong yolo_db_upgrade2_resize.py를 통해 (480, 640)을 (240, 320)으로 줄임 - 탐지 확인

 304b544733f9c8899d597f941069c3072990d024

pc_db와 pi_pc에 각각 연결하는 프로그램을 넣음 
2024.6.17 12.00 jinwoong pi에서 부하를 줄이는 작업 완료(이미지 축소, 흑백영상으로 변환, sleep시간) sender.py가 송신, receiver.py가 수신용 파이썬 프로그램(improved는 앞의 것에도 부하가 심할 때 사용)

2024.6.18 16:20 jinwoong receiver_pc_db_upgrade.py로 수신 가능하게 바꿈
