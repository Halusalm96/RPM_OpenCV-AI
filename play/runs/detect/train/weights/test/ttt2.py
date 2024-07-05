import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# TensorFlow Lite 모델 로드
interpreter = tf.lite.Interpreter(model_path='model2.tflite')
interpreter.allocate_tensors()

# 입력 및 출력 텐서 정보 가져오기
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 클래스 이름 목록
class_names = [
    "rollercoaster", "viking", "merrygoround", "ferriswheel"]

def preprocess_image(frame, input_shape):
    image = cv2.resize(frame, (input_shape[2], input_shape[3]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array(image).astype(np.float32)
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image

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
                class_probs = detection[5:5+num_classes]
                class_id = np.argmax(class_probs)
                class_confidence = class_probs[class_id]
                if class_confidence > best_confidence:
                    best_class_id = class_id
                    best_confidence = class_confidence
                    best_box = detection[:4]
    
    return best_class_id, best_confidence, best_box

def draw_box(frame, box, class_name, confidence):
    height, width, _ = frame.shape
    x_center, y_center, box_width, box_height = box
    x_center = x_center * width
    y_center = y_center * height
    box_width = box_width * width
    box_height = box_height * height
    
    x1 = int(x_center - (box_width / 2))
    y1 = int(y_center - (box_height / 2))
    x2 = int(x_center + (box_width / 2))
    y2 = int(y_center + (box_height / 2))
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"{class_name}: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 웹캠으로부터 프레임을 읽기 위한 초기화
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    input_shape = input_details[0]['shape']
    input_data = preprocess_image(frame, input_shape)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    best_class_id, best_confidence, best_box = parse_output(output_data)

    if best_class_id != -1 and best_confidence > 0.5:
        class_name = class_names[best_class_id]
        draw_box(frame, best_box, class_name, best_confidence)

    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
