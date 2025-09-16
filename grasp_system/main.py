import sys

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, Qt, QRect
from PyQt5.QtGui import QPixmap, QPainter, QColor
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout

from grasp_system.core.backend import SystemBackend
from grasp_system.ui.stylesheet import STYLE_SHEET
from grasp_system.ui.ui_components import ControlPanel, VisionLogPanel, AnalysisPanel
from grasp_system.utils.drawing_utils import DrawingUtils, format_image_for_display
from grasp_system.utils.logger_setup import setup_logging


class MainWindow(QMainWindow):
    """
    UI主窗口, 负责组装所有独立的UI面板。
    它的主要职责是作为“应用控制器(App Controller)”，连接UI和后端的信号与槽。
    """
    cleanup_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("文本驱动的视觉引导抓取系统")
        self.setGeometry(50, 50, 1400, 900)

        # --- 1. 初始化视图 (View) ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # 从 ui_components.py 中实例化各个面板
        self.control_panel = ControlPanel()
        self.vision_log_panel = VisionLogPanel()
        self.analysis_panel = AnalysisPanel()

        # 将面板添加到主布局中，保持比例，允许自由缩放
        main_layout.addWidget(self.control_panel, 2)
        main_layout.addWidget(self.vision_log_panel, 6)
        main_layout.addWidget(self.analysis_panel, 2)

        # --- 2. 初始化模型 (Model/Backend) & 线程 ---
        self.backend = SystemBackend()
        self.backend_thread = QThread()
        self.backend.moveToThread(self.backend_thread)

        # --- 3. 初始化控制器 (Controller): 连接所有信号和槽 ---
        self._setup_connections()

        # --- 4. 启动后端线程 ---
        self.backend_thread.start()

    def _setup_connections(self):
        """连接所有信号和槽。"""
        # --- 后端 -> UI ---
        self.backend.log_signal.connect(self.vision_log_panel.log_box.append)
        self.backend.device_connection_signal.connect(self.control_panel.update_device_status)
        self.backend.main_image_signal.connect(self._update_main_image)
        self.backend.roi_image_signal.connect(self._update_roi_image)
        self.backend.quality_map_signal.connect(self._update_quality_map)
        self.backend.angle_map_signal.connect(self._update_angle_map)
        self.backend.width_map_signal.connect(self._update_width_map)
        self.backend.grasp_enable_signal.connect(self.control_panel.execute_button.setEnabled)

        # --- UI -> 后端 ---
        cp = self.control_panel
        cp.connect_camera_signal.connect(self.backend.connect_camera)
        cp.disconnect_camera_signal.connect(self.backend.disconnect_camera)
        cp.connect_arm_signal.connect(self.backend.connect_arm)
        cp.disconnect_arm_signal.connect(self.backend.disconnect_arm)
        cp.connect_gripper_signal.connect(self.backend.connect_gripper)
        cp.disconnect_gripper_signal.connect(self.backend.disconnect_gripper)
        cp.instruction_signal.connect(self.backend.process_instruction)
        cp.strategy_signal.connect(self.backend.set_selection_strategy)
        cp.execute_grasp_signal.connect(self.backend.execute_grasp)
        cp.stop_signal.connect(self.backend.stop_all_tasks)
        # --- 新增: 连接模式切换信号 ---
        cp.mode_changed_signal.connect(self.backend.set_mode)
        self.cleanup_signal.connect(self.backend.cleanup)

        cp.detection_toggle_signal.connect(self.backend.set_detection_enabled)
        cp.grasp_toggle_signal.connect(self.backend.set_grasp_enabled)

        # --- 线程管理 ---
        self.backend_thread.started.connect(self.backend.run)
        self.backend.finished.connect(self.backend_thread.quit)
        self.backend.finished.connect(self.backend.deleteLater)
        self.backend_thread.finished.connect(self.backend_thread.deleteLater)
        print("All signal-slot connections established.")

    # --- 负责接收后端数据并更新UI ---
    @pyqtSlot(dict)
    def _update_main_image(self, data: dict):
        """更新主视觉区，并将绘制任务委托给 DrawingUtils。"""
        frame = data.get('frame')

        label = self.vision_log_panel.main_video_label
        canvas = QPixmap(label.size())
        canvas.fill(QColor('black'))

        video_pixmap = format_image_for_display(frame)

        if video_pixmap:
            painter = QPainter(canvas)
            video_size = video_pixmap.size()
            x = (label.width() - video_size.width()) // 2
            y = (label.height() - video_size.height()) // 2
            target_rect = QRect(x, y, video_size.width(), video_size.height())

            painter.drawPixmap(target_rect, video_pixmap)

            # 将绘制任务委托给 DrawingUtils
            painter.translate(x, y)  # 平移坐标系到视频帧左上角
            DrawingUtils.draw_all_annotations(painter, data)
            painter.end()

        label.setPixmap(canvas)

    @pyqtSlot(object)
    def _update_roi_image(self, roi):
        """更新分析区的ROI图像，进行缩放以适应空间。"""
        pixmap = format_image_for_display(roi)
        if pixmap:
            self.analysis_panel.roi_display.setPixmap(
                pixmap.scaled(self.analysis_panel.roi_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    @pyqtSlot(object)
    def _update_quality_map(self, q_img):
        """
        优化：使用固定的 [0, 1] 范围来应用色彩图，与matplotlib的 vmin/vmax 一致。
        这确保了质量分数的颜色在不同帧之间具有可比性。
        """
        if q_img is None: return

        # 1. 将q_img的值裁剪到[0, 1]范围，并缩放到[0, 255]
        # np.clip确保超出范围的值被修正
        q_img_normalized = np.clip(q_img, 0, 1) * 255

        # 2. 转换为uint8并应用色彩图
        heatmap = cv2.applyColorMap(
            q_img_normalized.astype(np.uint8),
            cv2.COLORMAP_JET
        )

        pixmap = format_image_for_display(heatmap)
        if pixmap:
            self.analysis_panel.q_display.setPixmap(
                pixmap.scaled(self.analysis_panel.q_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    @pyqtSlot(object)
    def _update_angle_map(self, ang_img):
        """
        优化：代码逻辑已正确实现matplotlib的 hsv 色彩图效果，添加注释说明。
        ang_img 的范围是 [-pi/2, pi/2]，通过归一化映射到HSV的色调(Hue)通道。
        """
        if ang_img is None: return

        # 1. 将角度从 [-pi/2, pi/2] 归一化到 [0, 1]
        normalized_angle = (ang_img + np.pi / 2) / np.pi

        # 2. 创建一个HSV图像
        #   - 色调(H)通道: 由归一化后的角度决定 (OpenCV中H范围是0-179)
        #   - 饱和度(S)通道: 设为最大值255，表示颜色最纯
        #   - 亮度(V)通道: 设为最大值255，表示颜色最亮
        hsv_img = np.zeros((*ang_img.shape, 3), dtype=np.uint8)
        hsv_img[..., 0] = (normalized_angle * 180).astype(np.uint8)
        hsv_img[..., 1] = 255
        hsv_img[..., 2] = 255

        # 3. 将HSV图像转换为RGB以便显示
        rgb_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)

        pixmap = format_image_for_display(rgb_img)
        if pixmap:
            self.analysis_panel.ang_display.setPixmap(
                pixmap.scaled(self.analysis_panel.ang_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    @pyqtSlot(object)
    def _update_width_map(self, width_img):
        """
        优化：使用固定的 [0, 100] 范围来应用色彩图，与matplotlib的 vmin/vmax 一致。
        这为抓取宽度提供了一个固定的、可比较的视觉标尺。
        """
        if width_img is None: return

        # 1. 将width_img的值裁剪到[0, 100]范围
        width_img_clipped = np.clip(width_img, 0, 100)

        # 2. 将裁剪后的值从[0, 100]线性映射到[0, 255]
        width_img_normalized = (width_img_clipped / 100.0) * 255

        # 3. 转换为uint8并应用色彩图
        heatmap = cv2.applyColorMap(
            width_img_normalized.astype(np.uint8),
            cv2.COLORMAP_JET
        )

        pixmap = format_image_for_display(heatmap)
        if pixmap:
            self.analysis_panel.width_display.setPixmap(
                pixmap.scaled(self.analysis_panel.width_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def closeEvent(self, event):
        """关闭窗口时，请求后端进行清理并等待线程安全退出。"""
        print("Main window is closing...")
        self.cleanup_signal.emit()
        if hasattr(self, 'backend_thread') and not self.backend_thread.wait(3000):
            print("Warning: Backend thread did not terminate gracefully.")
        event.accept()


class AppController:
    """负责创建和管理UI、后端和线程的生命周期"""

    def __init__(self, app):
        self.app = app
        self.window = MainWindow()
        self.backend = self.window.backend  # 从MainWindow获取backend实例的引用
        self.backend_thread = self.window.backend_thread  # 获取线程引用

        self._setup_app_connections()

    def _setup_app_connections(self):
        # 应用程序级别的连接，主要是处理退出
        self.app.aboutToQuit.connect(self._on_app_quit)

    def run(self):
        self.window.show()

    def _on_app_quit(self):
        print("Application is about to quit.")
        # 确保在应用退出前，清理信号被发射
        self.window.cleanup_signal.emit()


def main():
    """
    应用程序的入口函数。
    """
    setup_logging()

    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE_SHEET)

    # MainWindow内部已经处理了Backend和Thread的创建及基础连接
    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
