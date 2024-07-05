import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("yolov8n_model_directory")

tflite_model = converter.convert()

with open ("yolov8n.tflite", "wb") as f :
    f.write(tflite_model)