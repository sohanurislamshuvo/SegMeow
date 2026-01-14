from ultralytics import YOLO

def train_model(data_yaml, weights, epochs, imgsz):
    model = YOLO(weights)
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz
    )
    return model
