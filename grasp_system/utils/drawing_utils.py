# -*- coding: utf-8 -*-
# @Time       : 2025/9/9 23:06
# @File       : drawing_utils.py
# @Description: 包含了所有可视化标注的绘制工具
# grasp_system/drawing_utils.py
import cv2
import numpy as np
from PyQt5.QtCore import QPointF, Qt, QRectF
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QPolygonF, QImage, QPixmap, QBrush


class DrawingUtils:
    """
    一个用于在 QPainter 画布上绘制各种可视化标注的工具类。
    所有方法都是静态的，可以直接通过类名调用。
    """
    # --- 统一定义样式常量 ---
    DETECTION_COLOR = QColor("#A3BE8C")  # 绿色
    GRASP_COLOR = QColor("#BF616A")  # 红色
    TEXT_COLOR = QColor("#ECEFF4")  # 浅灰白色

    DETECTION_PEN = QPen(DETECTION_COLOR, 1, Qt.SolidLine)
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
            points_xy = grasp.get('points')  # (x, y) 顺序
            quality = grasp.get('quality', 0.0)  # 抓取置信度
            angle = grasp.get('angle', 0)  # 抓取角度
            center = grasp.get('center', [])  # 中心点坐标 (x, y) 顺序
            if not points_xy or len(points_xy) != 4:
                continue

            # QPointF 需要(x, y)顺序
            q_points = [QPointF(p[0], p[1]) for p in points_xy]
            polygon = QPolygonF(q_points)

            # 绘制矩形框
            painter.setBrush(Qt.NoBrush)
            painter.setPen(DrawingUtils.GRASP_PEN)
            painter.drawPolygon(polygon)

            # 绘制中心点（红色小点）
            if center and len(center) == 2:
                # 保存当前画笔设置
                current_pen = painter.pen()
                current_brush = painter.brush()

                # 设置红色画笔和画刷
                painter.setPen(QPen(Qt.red, 1))
                painter.setBrush(QBrush(Qt.red))

                # 绘制小圆作为中心点（半径2像素）
                center_point = QPointF(center[0], center[1])
                painter.drawEllipse(center_point, 2, 2)

                # 恢复原始画笔设置
                painter.setPen(current_pen)
                painter.setBrush(current_brush)

            # 计算矩形的中心和底部位置
            center_x = sum(p.x() for p in q_points) / 4
            # 找到矩形的底部y坐标（最大y值）
            bottom_y = max(p.y() for p in q_points)

            # 准备要显示的文本
            quality_text = f"q={quality:.2f} angle={angle:.2f}"

            # 设置字体，确保中文正常显示
            font = DrawingUtils.SCORE_FONT
            font.setFamily("SimHei")  # 使用黑体显示中文
            painter.setFont(font)
            painter.setPen(DrawingUtils.TEXT_PEN)

            # 计算文本尺寸
            text_rect = painter.fontMetrics().boundingRect(quality_text)

            # 将文本放在矩形正下方，中心对齐
            # 文本顶部与矩形底部保持一定距离（3像素）
            text_x = center_x - text_rect.width() / 2
            text_y = bottom_y + 3  # 3像素的间距
            text_rect.moveTo(int(text_x), int(text_y))

            # 绘制文本
            painter.drawText(text_rect, Qt.AlignLeft, quality_text)


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
