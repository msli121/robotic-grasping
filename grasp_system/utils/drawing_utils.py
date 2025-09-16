# -*- coding: utf-8 -*-
# @Time       : 2025/9/9 23:06
# @File       : drawing_utils.py
# @Description: 包含了所有可视化标注的绘制工具
# grasp_system/drawing_utils.py
import cv2
import numpy as np
from PyQt5.QtCore import QPointF, Qt, QRectF
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QPolygonF, QImage, QPixmap


class DrawingUtils:
    """
    一个用于在 QPainter 画布上绘制各种可视化标注的工具类。
    所有方法都是静态的，可以直接通过类名调用。
    """
    # --- 统一定义样式常量 ---
    DETECTION_COLOR = QColor("#A3BE8C")  # 绿色
    GRASP_COLOR = QColor("#BF616A")  # 红色
    TEXT_COLOR = QColor("#ECEFF4")  # 浅灰白色

    DETECTION_PEN = QPen(DETECTION_COLOR, 2, Qt.SolidLine)
    GRASP_PEN = QPen(GRASP_COLOR, 2, Qt.SolidLine)
    TEXT_PEN = QPen(TEXT_COLOR, 1, Qt.SolidLine)

    LABEL_FONT = QFont("微软雅黑", 12, QFont.Bold)
    SCORE_FONT = QFont("Arial", 10)

    @staticmethod
    def draw_all_annotations(painter: QPainter, display_data: dict):
        """
        绘制所有标注的总入口
        :param painter: QPainter 对象。
        :param display_data: 包含帧和标注信息的字典。
        """
        detections = display_data.get('detections', [])
        grasps = display_data.get('grasps', [])

        if detections:
            DrawingUtils.draw_detections(painter, detections)

        if grasps:
            DrawingUtils.draw_grasp_predictions(painter, grasps)

    @staticmethod
    def draw_detections(painter: QPainter, detections: list):
        """
        绘制所有目标检测框、中文类别名和置信度。
        """
        for det in detections:
            bbox = det.get('bbox')
            # 假设检测结果中没有中文标签，我们简化显示
            label = det.get('class_name', '')
            score = det.get('score', 0.0)
            if not bbox:
                continue

            x1, y1, x2, y2 = bbox
            rect = QRectF(x1, y1, x2 - x1, y2 - y1)

            painter.setPen(DrawingUtils.DETECTION_PEN)
            painter.drawRect(rect)

            text = f"{label}:{score:.2f}"
            painter.setFont(DrawingUtils.LABEL_FONT)

            text_rect = painter.fontMetrics().boundingRect(text)
            p_top_left = QPointF(int(x1), int(y1) - text_rect.height() - 2).toPoint()
            text_rect.moveTopLeft(p_top_left)

            # painter.setBrush(QColor(46, 52, 64, 180))  # 半透明背景
            # painter.setPen(Qt.NoPen)
            # painter.drawRect(text_rect.adjusted(-2, -2, 2, 2))

            painter.setPen(DrawingUtils.TEXT_PEN)
            painter.drawText(text_rect, Qt.AlignCenter, text)

    @staticmethod
    def draw_grasp_predictions(painter: QPainter, grasps: list):
        """
        绘制所有抓取预测结果，包括由四个点构成的矩形和抓取质量。
        """
        for grasp in grasps:
            points_yx = grasp.get('points')  # 后端传来的是 (y, x) 顺序
            quality = grasp.get('quality', 0.0)
            if not points_yx or len(points_yx) != 4:
                continue

            # 需要 QPointF(x, y) 顺序
            q_points = [QPointF(p[0], p[1]) for p in points_yx]
            polygon = QPolygonF(q_points)

            painter.setBrush(Qt.NoBrush)
            painter.setPen(DrawingUtils.GRASP_PEN)
            painter.drawPolygon(polygon)

            center_x = sum(p.x() for p in q_points) / 4
            center_y = sum(p.y() for p in q_points) / 4

            quality_text = f"{quality:.2f}"
            painter.setFont(DrawingUtils.SCORE_FONT)
            painter.setPen(DrawingUtils.TEXT_PEN)

            text_rect = painter.fontMetrics().boundingRect(quality_text)
            text_rect.moveCenter(QPointF(center_x, center_y).toPoint())
            painter.drawText(text_rect, Qt.AlignCenter, quality_text)


def format_image_for_display(img: np.ndarray | None) -> QPixmap | None:
    """
    将Numpy图像数组安全地转换为QPixmap。
    - 处理None输入。
    - 处理灰度图和彩色图。
    - 对非uint8类型进行健壮的归一化。
    - **修复: 移除.rgbSwapped()以正确显示颜色。**
    """
    if img is None or img.size == 0:
        return None

    img_copy = img.copy()

    if len(img_copy.shape) == 2:
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2RGB)

    if img_copy.dtype != np.uint8:
        if np.max(img_copy) <= 1.0 and np.min(img_copy) >= 0.0:
            img_copy = (img_copy * 255).astype(np.uint8)
        else:
            img_copy = cv2.normalize(img_copy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    h, w, ch = img_copy.shape
    bytes_per_line = ch * w
    q_img = QImage(img_copy.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(q_img)
