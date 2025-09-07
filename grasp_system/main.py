# main.py (V3.9 - 最终美化版)

import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QLabel
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, Qt, QRect
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor

# 确保其他模块文件与 main.py 在同一个文件夹或Python路径下
from backend import SystemBackend
from stylesheet import STYLE_SHEET
from ui_components import ControlPanel, VisionLogPanel, AnalysisPanel


def format_image_for_display(img):
    """
    将Numpy图像数组安全地转换为QPixmap。
    - 处理None输入。
    - 处理灰度图和彩色图。
    - 对非uint8类型进行健壮的归一化。
    - **修复: 移除.rgbSwapped()以正确显示颜色。**
    """
    if img is None or img.size == 0:
        # 返回None，让调用者决定如何处理空图像
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
        self.cleanup_signal.connect(self.backend.cleanup)

        # --- 线程管理 ---
        self.backend_thread.started.connect(self.backend.run)
        self.backend.finished.connect(self.backend_thread.quit)
        self.backend.finished.connect(self.backend.deleteLater)
        self.backend_thread.finished.connect(self.backend_thread.deleteLater)
        print("All signal-slot connections established.")

    # --- 私有槽函数, 负责接收后端数据并更新UI ---
    @pyqtSlot(object)
    def _update_main_image(self, frame):
        """
        核心优化: 更新主视觉区。
        保持视频流内容(640x480)的原始分辨率，并将其居中绘制在可缩放的QLabel上。
        """
        # 1. 获取 QLabel 当前的尺寸 (这是可变的布局空间)
        label = self.vision_log_panel.main_video_label
        canvas = QPixmap(label.size())
        canvas.fill(QColor('black'))  # 创建一个与QLabel等大的黑色画布

        # 2. 将传入的视频帧 (numpy array) 转换为 QPixmap
        video_pixmap = format_image_for_display(frame)

        # 3. 如果成功转换 (帧不是空的)
        if video_pixmap:
            # 4. 创建一个 QPainter 在我们的画布上进行绘制
            painter = QPainter(canvas)

            # 5. 计算目标绘制区域，使其在画布中央
            video_size = video_pixmap.size()  # 这是固定的 640x480
            x = (label.width() - video_size.width()) // 2
            y = (label.height() - video_size.height()) // 2
            target_rect = QRect(x, y, video_size.width(), video_size.height())

            # 6. 将视频流 QPixmap 绘制到画布的中央
            painter.drawPixmap(target_rect, video_pixmap)
            painter.end()

        # 7. 将最终绘制好的画布(可能带有黑边)设置为 QLabel 的内容
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
        heatmap = cv2.applyColorMap(
            cv2.normalize(q_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        pixmap = format_image_for_display(heatmap)
        if pixmap:
            self.analysis_panel.q_display.setPixmap(
                pixmap.scaled(self.analysis_panel.q_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    @pyqtSlot(object)
    def _update_angle_map(self, ang_img):
        hsv_img = np.zeros((*ang_img.shape, 3), dtype=np.uint8)
        normalized_angle = (ang_img + np.pi / 2) / np.pi
        hsv_img[..., 0] = (normalized_angle * 180).astype(np.uint8)
        hsv_img[..., 1] = 255
        hsv_img[..., 2] = 255
        rgb_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
        pixmap = format_image_for_display(rgb_img)
        if pixmap:
            self.analysis_panel.ang_display.setPixmap(
                pixmap.scaled(self.analysis_panel.ang_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    @pyqtSlot(object)
    def _update_width_map(self, width_img):
        pixmap = format_image_for_display(width_img)
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
    """负责创建和管理UI、后端和线程的生命周期。"""

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
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE_SHEET)

    # MainWindow内部已经处理了Backend和Thread的创建及基础连接
    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
