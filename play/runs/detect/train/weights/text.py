from ultralytics import YOLO # type: ignore

model = YOLO(model='last.pt')
model.export(format="onnx")