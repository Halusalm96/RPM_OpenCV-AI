import onnx

onnx_model = onnx.load('model.onnx')

print("Model inputs:")
for input in onnx_model.graph.input:
    print(input.name)