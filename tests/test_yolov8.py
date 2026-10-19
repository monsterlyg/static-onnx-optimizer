from onnx_static_optimizer import optimize
from ultralytics import YOLO


if __name__ == "__main__":
     # Export the model to ONNX format
    # model.export(format="onnx", dynamic=True, simplify=False)
    # exit()
    result = optimize("yolov8n.onnx", {"images": [1, 3, 640, 640]})
    print(result.report())

    if result.is_static:
        result.save("yolov8n_static.onnx")