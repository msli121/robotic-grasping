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


def train_yoloe():
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
    # model = YOLOE(r"D:\PycharmProjects\robotic-grasping\yolo\runs\detect\train_yoloe_20250915_5\weights\best.pt")
    model = YOLOE(r"D:\PycharmProjects\robotic-grasping\yolo\pretrained_models\yoloe-11s-seg.pt")
    # model = YOLOE(r"D:\PycharmProjects\robotic-grasping\yolo\pretrained_models\yoloe-11s-seg-pf.pt")
    model.to('cuda:0')
    # Set text prompt to detect person and bus. You only need to do this once after you load the model.

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
            names = ["toothbrush", "crayon", "battery"]
            model.set_classes(names, model.get_text_pe(names))
            # model.set_classes(names)
            results = model.predict(bgr_frame, conf=0.2)

            # visualize
            # results[0].show()
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


def predict_yoloe_pf():
    model = YOLOE(r"D:\PycharmProjects\robotic-grasping\yolo\pretrained_models\yoloe-11s-seg-pf.pt")
    model.to('cuda:0')
    camera = RealSenseCamera()
    camera.connect()
    print("Camera connected.")
    print("\nStarting inference loop. Press 'q' in the window to quit.")

    try:
        while True:
            img_info = camera.get_image_bundle()
            rgb_frame = img_info.get('rgb')
            if rgb_frame is None:
                continue

            # RGB -> BGR
            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

            # 创建一个副本用于绘制，避免在原始图像上修改
            display_frame = bgr_frame.copy()

            # 推理
            results = model.predict(bgr_frame, conf=0.4)

            # --- 手动绘制检测结果，替代 results[0].plot() ---
            if results and results[0]:
                # 获取类别名称
                class_names = results[0].names
                # 遍历每一个检测到的物体
                for box in results[0].boxes:
                    # 获取边界框坐标 (x1, y1, x2, y2)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # 获取置信度
                    confidence = box.conf[0]
                    # 获取类别ID
                    class_id = int(box.cls[0])
                    # 定义颜色（这里用一个固定的颜色，也可以根据class_id生成不同颜色）
                    color = (0, 255, 0)  # 绿色
                    line_thickness = 2
                    # 1. 绘制边界框
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, line_thickness)
                    # 2. 准备标签文字
                    label = f'{class_names[class_id]} {confidence:.2f}'
                    # 3. 为标签文字添加背景
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    font_thickness = 1
                    # 获取文字的尺寸
                    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
                    # 计算背景框的坐标
                    label_bg_x2 = x1 + text_width
                    label_bg_y2 = y1 - text_height - baseline

                    # 绘制实心矩形作为背景
                    cv2.rectangle(display_frame, (x1, y1), (label_bg_x2, label_bg_y2), color, -1)  # -1表示填充

                    # 4. 绘制文字
                    cv2.putText(display_frame, label, (x1, y1 - baseline), font, font_scale, (0, 0, 0),
                                font_thickness)  # 黑色字体

            # 显示优化后的图像
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
    # predict_yoloe_pf()
