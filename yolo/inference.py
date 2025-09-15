# inference.py
import logging
import os
import time
from abc import ABC, abstractmethod

import cv2
import numpy as np
from ultralytics import YOLO, YOLOE

from hardware.camera import RealSenseCamera
from yolo.visualizer import Visualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# ============================================================================
# YOLODetector
# ============================================================================
class YOLODetector():
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        logger.info(f"Loading YOLOv8 model from: {self.model_path}")
        if self.model_path and os.path.exists(self.model_path):
            self.model = YOLO(self.model_path)
            logger.info("YOLOv8 model loaded successfully.")
        else:
            logger.error(f"YOLOv8 model file not found: {self.model_path}")
        return self.model

    def detect(self, image: np.ndarray, target_classes: list = None, threshold: float = 0.5) -> list:
        """
      对单张图像进行目标检测，并返回结构化的结果

        :param image: 输入图像 (NumPy array, BGR或RGB格式)。
        :param target_classes: (可选) 一个包含目标类别名称的列表，只返回这些类别的检测结果。
                               如果为 None，则返回所有检测到的类别。
        :param threshold: (可选) 置信度阈值，低于此值的检测结果将被忽略。
        :return:  {'bbox': (x1, y1, x2, y2), 'score': float, 'class_id': int, 'class_name': str}
        """
        results = self.model.predict(image, verbose=False)
        detections = []
        if not results:
            return detections

        for r in results:
            for box in r.boxes.cpu().numpy():
                score = float(box.conf[0])
                if score < threshold:
                    continue

                cls_id = int(box.cls[0])
                cls_name = r.names[cls_id]

                # 按目标类别过滤
                if target_classes is not None and cls_name not in target_classes:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detection_dict = {
                    'bbox': (x1, y1, x2, y2),
                    'score': score,
                    'class_id': cls_id,
                    'class_name': cls_name
                }
                detections.append(detection_dict)

        return detections


# ============================================================================
# YOLODetector
# ============================================================================
class YOLOEDetector():
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        logger.info(f"Loading YOLOE model from: {self.model_path}")
        if self.model_path and os.path.exists(self.model_path):
            self.model = YOLOE(self.model_path)
            logger.info("YOLOE model loaded successfully.")
        else:
            logger.error(f"YOLOE model file not found: {self.model_path}")
        return self.model

    def detect(self, image: np.ndarray, target_classes: list = None, threshold: float = 0.5) -> list:
        """
      对单张图像进行目标检测，并返回结构化的结果

        :param image: 输入图像 (NumPy array, BGR或RGB格式)。
        :param target_classes: (可选) 一个包含目标类别名称的列表，只返回这些类别的检测结果。
                               如果为 None，则返回所有检测到的类别。
        :param threshold: (可选) 置信度阈值，低于此值的检测结果将被忽略。
        :return:  {'bbox': (x1, y1, x2, y2), 'score': float, 'class_id': int, 'class_name': str}
        """
        results = self.model.predict(image, verbose=False)
        detections = []
        if not results:
            return detections

        for r in results:
            for box in r.boxes.cpu().numpy():
                score = float(box.conf[0])
                if score < threshold:
                    continue

                cls_id = int(box.cls[0])
                cls_name = r.names[cls_id]

                # 按目标类别过滤
                if target_classes is not None and cls_name not in target_classes:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detection_dict = {
                    'bbox': (x1, y1, x2, y2),
                    'score': score,
                    'class_id': cls_id,
                    'class_name': cls_name
                }
                detections.append(detection_dict)

        return detections


class YOLOETextPromptDetector():
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        logger.info(f"Loading YOLOE model from: {self.model_path}")
        if self.model_path and os.path.exists(self.model_path):
            self.model = YOLOE(self.model_path)
            logger.info("YOLOE model loaded successfully.")
        else:
            logger.error(f"YOLOE model file not found: {self.model_path}")
        return self.model

    def detect(self, image: np.ndarray, target_classes: list = None, threshold: float = 0.5) -> list:
        """
        对单张图像进行目标检测，并返回结构化的结果

        :param image: 输入图像 (NumPy array, BGR或RGB格式)。
        :param target_classes: (可选) 一个包含目标类别名称的列表，只返回这些类别的检测结果。
                               如果为 None，则返回所有检测到的类别。
        :param threshold: (可选) 置信度阈值，低于此值的检测结果将被忽略。
        :return:  {'bbox': (x1, y1, x2, y2), 'score': float, 'class_id': int, 'class_name': str}
        """
        self.model.set_classes(target_classes, self.model.get_text_pe(target_classes))
        results = self.model.predict(image, verbose=False)
        detections = []
        if not results:
            return detections
        for r in results:
            for box in r.boxes.cpu().numpy():
                score = float(box.conf[0])
                if score < threshold:
                    continue
                cls_id = int(box.cls[0])
                cls_name = r.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detection_dict = {
                    'bbox': (x1, y1, x2, y2),
                    'score': score,
                    'class_id': cls_id,
                    'class_name': cls_name
                }
                detections.append(detection_dict)
        return detections


# ============================================================================
# 独立验证函数
# ============================================================================
def visualize_camera_detection(model_path: str, font_path: str):
    """
    启动实时摄像头进行目标检测，并进行可视化显示。
    用于独立验证模型效果。

    :param model_path: YOLOv8 模型的路径。
    :param font_path: 用于显示中文标签的字体文件路径。
    """
    camera = None
    try:
        # 1. 初始化
        logger.info("Initializing components...")
        detector = YOLODetector(model_path=model_path)
        detector.load_model()

        visualizer = Visualizer(font_path=font_path)

        camera = RealSenseCamera()
        camera.connect()
        logger.info("Camera connected.")
        logger.info("\nStarting inference loop. Press 'q' in the window to quit.")

        # 2. 实时检测与显示循环
        prev_time = time.time()
        while True:
            img_info = camera.get_image_bundle()
            rgb_frame = img_info.get('rgb')
            if rgb_frame is None:
                continue

            # 使用YOLOv8_Detector的内部方法，返回适合可视化的格式
            detections_raw = detector.detect(rgb_frame)

            # 将摄像头获取的 RGB 图像转换为 OpenCV 使用的 BGR 格式
            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            display_frame = visualizer.draw_detections(bgr_frame, detections_raw, confidence_threshold=0.6)

            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            display_frame = visualizer.draw_fps(display_frame, fps)

            cv2.imshow('YOLOv8 Real-time Detection', display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        # 3. 清理资源
        logger.info("Cleaning up...")
        cv2.destroyAllWindows()
        if camera:
            camera.disconnect()
        logger.info("Cleanup complete. Exiting.")


if __name__ == '__main__':
    MODEL_PATH = r"D:\PycharmProjects\robotic-grasping\yolov8\runs\detect\train_20250914_144614\weights\best.pt"
    # 字体文件路径，或者更换为其他中文字体路径
    FONT_PATH = "C:/Windows/Fonts/simhei.ttf"
    if not os.path.isfile(MODEL_PATH):
        logger.error(f"Error: Model file not found at '{MODEL_PATH}'.")
        exit()
    if not os.path.exists(FONT_PATH):
        logger.error(f"Error: Font file not found at '{FONT_PATH}'.")
        exit()

    # 可视化验证
    visualize_camera_detection(model_path=MODEL_PATH, font_path=FONT_PATH)
