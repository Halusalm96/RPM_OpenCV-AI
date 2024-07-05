from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("best.pt")

# Export the model to TFLite format
model.export(format="tflite")  # creates 'yolov8n_float32.tflite'

# Load the exported TFLite model
tflite_model = YOLO("best.tflite")

# Run inference 
tflite_model.predict('test_images', save=True, imgsz=640, conf=0.2)