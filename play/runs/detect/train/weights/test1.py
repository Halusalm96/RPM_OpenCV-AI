from onnx_tf.backend import prepare
import onnx

# onnx 파일 -> TensorFlow 파일로 변환
onnx_model = onnx.load("yolov8n.onnx")

tf_rep = prepare(onnx_model)

tf_rep.export_graph("yolov8n_model_directory")