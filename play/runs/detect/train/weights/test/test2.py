import tensorflow as tf

# TensorFlow Lite 변환기 생성
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')

# 기본 양자화 옵션 설정
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 양자화 수행을 위한 추가 설정 (필요시 주석 해제)
# 양자화 인식 학습(QAT) 모델을 사용하는 경우
# converter.representative_dataset = representative_dataset_gen

# 변환 수행
tflite_model = converter.convert()

# TFLite 모델 저장
with open('model2.tflite', 'wb') as f:
    f.write(tflite_model)
