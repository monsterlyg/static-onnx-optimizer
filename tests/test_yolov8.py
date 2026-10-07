from onnx_static_optimizer import optimize


if __name__ == "__main__":
    result = optimize("yolov8n.onnx", {"images": [1, 3, 640, 640]})
    print(result.report())

    if result.is_static:
        result.save("yolov8n_static.onnx")