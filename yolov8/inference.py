# inference.py (支持中文显示的最终工程实践版 V4)

import os
import time
from abc import ABC, abstractmethod

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from hardware.camera import RealSenseCamera


# ============================================================================
# 1. 策略定义
# ============================================================================
class DetectionStrategy(ABC):
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        self.model_path = model_path
        self.model = None

    @abstractmethod
    def load_model(self): pass

    @abstractmethod
    def detect(self, image: np.ndarray, **kwargs) -> list: pass


# ============================================================================
# 2. YOLOv8_Detector
# ============================================================================
class YOLOv8_Detector(DetectionStrategy):
    def __init__(self, model_path):
        super().__init__(model_path)
        self.model_path = model_path

    def load_model(self):
        print(f"Loading YOLOv8 model from: {self.model_path}")
        self.model = YOLO(self.model_path)
        print("YOLOv8 model loaded successfully.")

    def detect(self, image: np.ndarray, **kwargs) -> list:
        if self.model is None:
            print("Error: Model is not loaded.")
            raise ValueError("Model is not loaded. Please call load_model() first.")
        results = self.model.predict(image)
        detections = []
        if results:
            for r in results:
                for box in r.boxes.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    score, cls_id = float(box.conf[0]), int(box.cls[0])
                    cls_name = r.names[cls_id]
                    detections.append(((x1, y1, x2, y2), score, cls_id, cls_name))
        return detections


# ============================================================================
# 3. 可视化工具类
# ============================================================================
class Visualizer:
    """封装所有绘图逻辑的工具类, 使用Pillow支持中文。"""

    def __init__(self, font_path=None):
        # 颜色为 BGR (OpenCV) 或 RGB (Pillow)
        self.box_color_bgr = (0, 255, 0)
        self.box_color_rgb = (0, 255, 0)
        self.box_thickness = 2

        # 字体相关
        self.font_path = font_path
        self.font_size = 15
        self.font = self._load_font()

    def _load_font(self):
        """加载字体文件，如果失败则使用Pillow默认字体。"""
        if self.font_path and os.path.exists(self.font_path):
            try:
                return ImageFont.truetype(self.font_path, self.font_size)
            except IOError:
                print(f"Warning: Font file '{self.font_path}' could not be loaded. Using default font.")
        else:
            print("Warning: Font path not provided or not found. Using default font.")
        return ImageFont.load_default()

    def draw_detections(self, image: np.ndarray, detections: list, confidence_threshold=0.5):
        """
        在图像上绘制检测结果，支持中文。
        :param image: BGR格式的OpenCV图像。
        """
        # 1. 转换到Pillow图像进行文本和矩形绘制
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_image)
        draw = ImageDraw.Draw(pil_img)

        for (bbox, score, _, class_name) in detections:
            if score < confidence_threshold:
                continue

            x1, y1, x2, y2 = bbox

            # --- 绘制绿色实线框 ---
            draw.rectangle([x1, y1, x2, y2], outline=self.box_color_rgb, width=self.box_thickness)

            # --- 准备标签文本 ---
            label = f"{class_name} {score:.2f}"

            # --- 使用Pillow绘制带背景的中文文本 ---
            try:
                # Pillow >= 10.0.0
                text_bbox = draw.textbbox((x1, y1), label, font=self.font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
            except AttributeError:
                # Pillow < 10.0.0 (兼容旧版)
                text_width, text_height = draw.textsize(label, font=self.font)

            # 绘制背景
            background_bbox = [x1, y1 - text_height - 4, x1 + text_width + 4, y1]
            draw.rectangle(background_bbox, fill=self.box_color_rgb)

            # 绘制文本
            draw.text((x1 + 2, y1 - text_height - 2), label, font=self.font, fill=(0, 0, 0))

        # 2. 转换回OpenCV图像
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def draw_fps(self, image: np.ndarray, fps: float):
        """在图像右上角绘制FPS (使用OpenCV)。"""
        display_image = image.copy()
        fps_text = f"FPS: {fps:.2f}"
        font_cv = cv2.FONT_HERSHEY_SIMPLEX

        (text_width, text_height), _ = cv2.getTextSize(fps_text, font_cv, 0.6, 2)

        cv2.rectangle(display_image, (image.shape[1] - text_width - 20, 10), (image.shape[1] - 10, 20 + text_height),
                      (0, 0, 0), cv2.FILLED)
        cv2.putText(display_image, fps_text, (image.shape[1] - text_width - 15, 15 + text_height), font_cv, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
        return display_image


# ============================================================================
# 4. 主执行逻辑 (最终封装版)
# ============================================================================
class InferenceApp:
    def __init__(self, model_path, font_path):
        self.model_path = model_path

        self.detector = None
        self.camera = None
        self.visualizer = Visualizer(font_path=font_path)

        self.is_running = True

    def initialize(self):
        print("Initializing components...")
        try:
            self.detector = YOLOv8_Detector(model_path=self.model_path)
            self.detector.load_model()

            self.camera = RealSenseCamera()
            self.camera.connect()
            print("Camera connected.")
            return True
        except Exception as e:
            print(f"Initialization failed: {e}")
            return False

    def run_inference_loop(self):
        print("\nStarting inference loop. Press 'q' to quit.")
        prev_time = time.time()

        while self.is_running:
            try:
                img_info = self.camera.get_image_bundle()
                rgb_frame = img_info.get('rgb')
                if rgb_frame is None: continue

                detections = self.detector.detect(rgb_frame)

                bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

                # 使用 Visualizer 绘制检测结果
                display_frame = self.visualizer.draw_detections(bgr_frame, detections, confidence_threshold=0.6)

                current_time = time.time()
                fps = 1 / (current_time - prev_time)
                prev_time = current_time

                # 使用 Visualizer 绘制 FPS
                display_frame = self.visualizer.draw_fps(display_frame, fps)

                cv2.imshow('YOLOv8 Inference', display_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.is_running = False

            except Exception as e:
                print(f"An error occurred in the main loop: {e}")
                self.is_running = False

    def cleanup(self):
        print("Cleaning up...")
        cv2.destroyAllWindows()
        if self.camera: self.camera.disconnect()
        print("Cleanup complete. Exiting.")


if __name__ == '__main__':
    MODEL_PATH = r"D:\PycharmProjects\robotic-grasping\yolov8\runs\detect\train2\weights\best.pt"
    # 中文字体路径 Windows 黑体
    FONT_PATH = "C:/Windows/Fonts/simhei.ttf"

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(MODEL_PATH)

    if not os.path.exists(FONT_PATH):
        print(f"Error: Font file not found at '{FONT_PATH}'.")
        print("Please download a Chinese font (e.g., simhei.ttf) or update the FONT_PATH.")
        exit()

    # 运行应用
    app = InferenceApp(
        model_path=MODEL_PATH,
        font_path=FONT_PATH,
    )

    if app.initialize():
        app.run_inference_loop()

    app.cleanup()
