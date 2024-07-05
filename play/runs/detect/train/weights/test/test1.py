import onnx
from onnx_tf.backend import prepare

# 정렬된 ONNX 모델 로드
onnx_model = onnx.load('model_sorted.onnx')

# TensorFlow 모델로 변환
tf_rep = prepare(onnx_model)
tf_rep.export_graph('saved_model')
