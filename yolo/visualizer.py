# -*- coding: utf-8 -*-
# @Time       : 2025/9/9 23:57
# @File       : visualizer.py
# @Description:
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class Visualizer:
    """封装所有绘图逻辑的工具类, 使用Pillow支持中文。"""

    def __init__(self, font_path=None):
        self.box_color_bgr = (0, 255, 0)
        self.box_color_rgb = (0, 255, 0)
        self.box_thickness = 2
        self.font_path = font_path
        self.font_size = 15
        self.font = self._load_font()

    def _load_font(self):
        if self.font_path and os.path.exists(self.font_path):
            try:
                return ImageFont.truetype(self.font_path, self.font_size)
            except IOError:
                print(f"Warning: Font file '{self.font_path}' could not be loaded. Using default font.")
        else:
            print("Warning: Font path not provided or not found. Using default font.")
        return ImageFont.load_default()

    def draw_detections(self, image: np.ndarray, detections: list, confidence_threshold=0.5):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_image)
        draw = ImageDraw.Draw(pil_img)
        for detection_item in detections:
            bbox = detection_item.get('bbox')
            score = detection_item.get('score')
            class_name = detection_item.get('class_name')
            class_id = detection_item.get('class_id')
            if score < confidence_threshold:
                continue
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline=self.box_color_rgb, width=self.box_thickness)
            label = f"{class_name} {score:.2f}"
            try:
                text_bbox = draw.textbbox((x1, y1), label, font=self.font)
                text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            except AttributeError:
                text_width, text_height = draw.textsize(label, font=self.font)
            background_bbox = [x1, y1 - text_height - 4, x1 + text_width + 4, y1]
            draw.rectangle(background_bbox, fill=self.box_color_rgb)
            draw.text((x1 + 2, y1 - text_height - 2), label, font=self.font, fill=(0, 0, 0))
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def draw_fps(self, image: np.ndarray, fps: float):
        display_image = image.copy()
        fps_text = f"FPS: {fps:.2f}"
        font_cv = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), _ = cv2.getTextSize(fps_text, font_cv, 0.6, 2)
        cv2.rectangle(display_image, (image.shape[1] - text_width - 20, 10), (image.shape[1] - 10, 20 + text_height),
                      (0, 0, 0), cv2.FILLED)
        cv2.putText(display_image, fps_text, (image.shape[1] - text_width - 15, 15 + text_height), font_cv, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
        return display_image
