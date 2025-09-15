# -*- coding: utf-8 -*-
# @Time       : 2025/9/14 12:49
# @File       : train_yoloe.py
# @Description:
import json
import time
import traceback

import cv2
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEPETrainer

from hardware.camera import RealSenseCamera

def train_yoloe_by_fine_tuning():
    # Initialize a detection model from a config
    # yaml_config = r'D:\PycharmProjects\robotic-grasping\yolo\model_config\yoloe-v8s.yaml'
    yaml_config = r'D:\PycharmProjects\robotic-grasping\yolo\model_config\yoloe-11.yaml'
    model = YOLOE(yaml_config)
    # model = YOLOE(r'D:\PycharmProjects\robotic-grasping\yolo\pretrained_models\yoloe-11s-seg.pt')

    # Load weights from a pretrained segmentation checkpoint (same scale)
    # pt_path = r'D:\PycharmProjects\robotic-grasping\yolo\pretrained_models\yolov8s.pt'
    pt_path = r'D:\PycharmProjects\robotic-grasping\yolo\pretrained_models\yoloe-11s-seg.pt'
    model.load(pt_path)

    dataset_yaml_path = r'D:\PycharmProjects\robotic-grasping\yolo\datasets\paper\dataset.yaml'
    # Fine-tune on your detection dataset
    now_str = time.strftime("%Y%m%d")
    results = model.train(
        data=dataset_yaml_path,  # Detection dataset
        epochs=60,
        patience=15,
        batch=16,
        save_period=10,
        name=f"train_yoloe_{now_str}_",
        trainer=YOLOEPETrainer,  # <- Important: use detection trainer
    )

    print("\nTraining completed.")

    # --- 4. 验证并记录结果 ---
    print("Running validation on the best model...")
    # `model` 对象在 train 后会自动加载最佳权重，所以可以直接调用 val()
    val_metrics = model.val()

    # 构造结果字典
    summary = {
        "dataset": dataset_yaml_path,
        "epochs_trained": 80,
        "batch_size": 16,
        "results_directory": str(results.save_dir),
        "best_model_path": str(results.save_dir / 'weights' / 'best.pt'),
        "metrics": {
            "mAP50": float(f"{val_metrics.box.map50:.4f}"),
            "mAP50-95": float(f"{val_metrics.box.map:.4f}"),
            "precision": float(f"{val_metrics.box.p.mean():.4f}"),
            "recall": float(f"{val_metrics.box.r.mean():.4f}")
        }
    }

    # 将摘要保存为 JSON 文件
    summary_path = results.save_dir / 'training_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print("\n--- Validation Metrics ---")
    for key, value in summary['metrics'].items():
        print(f"{key}: {value}")
    print(f"--------------------------")
    print(f"Best model saved at: {summary['best_model_path']}")
    print(f"Training summary saved to: {summary_path}")


def predict_yoloe():
    # Initialize a YOLOE model
    model = YOLOE(r"D:\PycharmProjects\robotic-grasping\yolo\runs\detect\train_yoloe_20250915_5\weights\best.pt")
    model.to('cuda:0')
    # Set text prompt to detect person and bus. You only need to do this once after you load the model.
    names = ["螺丝", "螺丝刀", "弹簧"]

    camera = RealSenseCamera()
    camera.connect()
    print("Camera connected.")
    print("\nStarting inference loop. Press 'q' in the window to quit.")

    try:
        while True:
            img_info = camera.get_image_bundle()
            rgb_frame = img_info.get('rgb')  # 原始 640x480 RGB 图像
            if rgb_frame is None:
                continue

            # RGB -> BGR
            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

            # inference
            model.set_classes(names, model.get_text_pe(names))
            # model.set_classes(names)
            results = model.predict(bgr_frame, conf=0.5)

            # visualize
            display_frame = results[0].plot()
            cv2.imshow('YOLOE Real-time Inference', display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        traceback.print_exc()
        print(f"An error occurred in the main loop: {e}")
    finally:
        cv2.destroyAllWindows()
        camera.disconnect()


if __name__ == '__main__':
    # train_yoloe()
    predict_yoloe()
