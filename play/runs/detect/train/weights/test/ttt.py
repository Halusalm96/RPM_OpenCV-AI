import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw

# TensorFlow Lite 모델 로드
interpreter = tf.lite.Interpreter(model_path='model2.tflite')
interpreter.allocate_tensors()

# 입력 및 출력 텐서 정보 가져오기
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 입력 텐서 정보 출력 (확인용)
print("Input Details:", input_details)

# 입력 이미지 로드 및 전처리
def load_image(image_path, input_shape):
    image = Image.open(image_path).resize((input_shape[2], input_shape[3]))  # [width, height]로 리사이즈
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = np.array(image).astype(np.float32)
    image = image / 255.0  # 정규화 (필요한 경우)
    image = np.transpose(image, (2, 0, 1))  # [channels, height, width]로 변환
    image = np.expand_dims(image, axis=0)  # 배치 차원 추가 [1, height, width, channels]
    return image

# 이미지 파일 경로
image_path = r'C:\Users\503\play\test\images\37d6f118bfa9668790f0145c699811ad.jpg'  # 실제 이미지 파일 경로로 변경

# 입력 이미지 로드
input_shape = input_details[0]['shape']
input_data = load_image(image_path, input_shape)

# 입력 데이터 설정
interpreter.set_tensor(input_details[0]['index'], input_data)

# 추론 실행
interpreter.invoke()

# 출력 데이터 가져오기
output_data = interpreter.get_tensor(output_details[0]['index'])

# 출력 데이터의 차원 확인
print(f"Output data shape: {output_data.shape}")

# 클래스 이름 목록
class_names = [
    "rollercoaster", "viking", "merrygoround", "ferriswheel"]

# 바운딩 박스와 클래스 확률 해석
def parse_output(output_data, threshold=0.5):
    best_class_id = -1
    best_confidence = -1
    best_box = None
    grid_size = output_data.shape[1]
    num_classes = len(class_names)
    
    for i in range(grid_size):
        for j in range(grid_size):
            detection = output_data[0, i, j, :]
            confidence = detection[4]
            if confidence > threshold:
                class_probs = detection[5:5+num_classes]  # 클래스 확률 추출
                class_id = np.argmax(class_probs)
                class_confidence = class_probs[class_id]
                if class_confidence > best_confidence:
                    best_class_id = class_id
                    best_confidence = class_confidence
                    best_box = detection[:4]
    
    return best_class_id, best_confidence, best_box

# 바운딩 박스 추출
best_class_id, best_confidence, best_box = parse_output(output_data)

# 바운딩 박스를 원본 이미지에 그리기
def draw_box(image_path, box, class_name, confidence):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    # 바운딩 박스 좌표 계산
    x_center, y_center, box_width, box_height = box
    x_center = x_center * width / 80
    y_center = y_center * height / 80
    box_width = box_width * width / 80
    box_height = box_height * height / 80
    
    x1 = x_center - (box_width / 2)
    y1 = y_center - (box_height / 2)
    x2 = x_center + (box_width / 2)
    y2 = y_center + (box_height / 2)
    
    # 디버깅 메시지 출력
    print(f"Drawing box: ({x1}, {y1}), ({x2}, {y2})")
    
    # 바운딩 박스 그리기
    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    draw.text((x1, y1 - 10), f"{class_name}: {confidence:.2f}", fill="red")  # 텍스트 위치 조정
    
    # 결과 이미지 저장
    result_image_path = 'result_image.jpg'
    image.save(result_image_path)
    print(f"Result image saved as {result_image_path}")

# 클래스 확률 출력 및 결과 이미지 저장
if best_class_id != -1 and best_confidence > 0.5:  # 0.5는 최소 신뢰도 기준입니다. 필요에 따라 조정 가능합니다.
    class_name = class_names[best_class_id]
    print(f"Class: {class_name}, Confidence: {best_confidence:.2f}")
    draw_box(image_path, best_box, class_name, best_confidence)
else:
    print("No high-confidence object detected.")
