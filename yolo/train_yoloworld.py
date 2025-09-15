import time

from ultralytics import YOLOWorld


def train():
    # Load a pretrained YOLOv8s-worldv2 model
    model_path = r'D:\PycharmProjects\robotic-grasping\yolo\pretrained_models\yolov8s-worldv2.pt'
    model = YOLOWorld(model_path)

    # Train the model on the COCO8 dataset for 100 epochs
    dataset_yaml_path = r'D:\PycharmProjects\robotic-grasping\yolo\datasets\paper\dataset.yaml'
    now_str = time.strftime("%Y%m%d")
    results = model.train(data=dataset_yaml_path,
                          epochs=100,
                          imgsz=640,
                          save_period=10,
                          name=f"train_yoloe_{now_str}_",
                          )


if __name__ == '__main__':
    train()
