# ui_components.py

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
                             QTextEdit, QComboBox, QRadioButton, QButtonGroup,
                             QFrame, QGridLayout, QSizePolicy, QCheckBox)
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt


class ControlPanel(QFrame):
    """左侧控制面板, 只负责UI元素的创建和布局, 通过信号向外发送用户操作。"""
    # 定义发出的信号
    connect_camera_signal = pyqtSignal()
    disconnect_camera_signal = pyqtSignal()
    connect_arm_signal = pyqtSignal()
    disconnect_arm_signal = pyqtSignal()
    connect_gripper_signal = pyqtSignal()
    disconnect_gripper_signal = pyqtSignal()

    instruction_signal = pyqtSignal(str)
    strategy_signal = pyqtSignal(str)
    execute_grasp_signal = pyqtSignal()
    stop_signal = pyqtSignal()

    # 识别和抓取开关信号
    detection_toggle_signal = pyqtSignal(bool)
    grasp_toggle_signal = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self._init_ui()
        self._setup_connections()
        self.update_mode_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # 设备连接
        device_frame = self._create_device_frame()
        # 任务控制
        task_frame = self._create_task_frame()
        # 执行控制
        exec_frame = self._create_exec_frame()

        layout.addWidget(device_frame)
        layout.addStretch()
        layout.addWidget(task_frame)
        layout.addStretch()
        layout.addWidget(exec_frame)
        layout.addStretch(4)

    def _create_device_frame(self):
        frame = QFrame()
        layout = QVBoxLayout(frame)
        layout.addWidget(QLabel("<b>设备连接</b>"))
        grid = QGridLayout()
        self.cam_status, self.cam_btns = self._create_device_row(grid, 0, "相机:")
        self.arm_status, self.arm_btns = self._create_device_row(grid, 1, "机械臂:")
        self.gripper_status, self.gripper_btns = self._create_device_row(grid, 2, "夹爪:")
        layout.addLayout(grid)
        return frame

    def _create_task_frame(self):
        frame = QFrame()
        layout = QVBoxLayout(frame)
        layout.addWidget(QLabel("<b>任务控制</b>"))

        # --- 新增功能开关 ---
        layout.addWidget(QLabel("功能开关:"))
        self.enable_detection_cb = QCheckBox("开启目标识别")
        self.enable_grasp_cb = QCheckBox("开启抓取预测")
        self.enable_detection_cb.setEnabled(False)
        self.enable_grasp_cb.setEnabled(False)
        switch_hbox = QHBoxLayout()
        switch_hbox.addWidget(self.enable_detection_cb)
        switch_hbox.addWidget(self.enable_grasp_cb)
        layout.addLayout(switch_hbox)
        # --- 结束 ---

        self.auto_mode_radio = QRadioButton("自动模式")
        self.inst_mode_radio = QRadioButton("指令模式")
        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.auto_mode_radio)
        self.mode_group.addButton(self.inst_mode_radio)
        self.inst_mode_radio.setChecked(True)
        mode_hbox = QHBoxLayout()
        mode_hbox.addWidget(self.auto_mode_radio)
        mode_hbox.addWidget(self.inst_mode_radio)
        layout.addLayout(mode_hbox)
        layout.addWidget(QLabel("目标选择策略:"))
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["置信度最高", "抓取最近的", "从左到右"])
        layout.addWidget(self.strategy_combo)
        self.auto_mode_button = QPushButton("开始/停止 自动模式")
        self.instruction_text = QTextEdit()
        self.instruction_text.setPlaceholderText("在此输入指令...")
        self.send_inst_button = QPushButton("发送指令")
        layout.addWidget(self.auto_mode_button)
        layout.addWidget(self.instruction_text)
        layout.addWidget(self.send_inst_button)
        return frame

    def _create_exec_frame(self):
        frame = QFrame()
        layout = QVBoxLayout(frame)
        self.execute_button = QPushButton("执行抓取")
        self.execute_button.setEnabled(False)
        self.stop_button = QPushButton("强制停止")
        layout.addWidget(self.execute_button)
        layout.addWidget(self.stop_button)
        return frame

    def _create_device_row(self, grid_layout, row, name):
        label = QLabel(name)
        indicator = QLabel()
        indicator.setObjectName("status_indicator")
        conn_btn = QPushButton("连接")
        disconn_btn = QPushButton("断开")
        conn_btn.setStyleSheet("padding: 5px")
        disconn_btn.setStyleSheet("padding: 5px")
        disconn_btn.setEnabled(False)
        grid_layout.addWidget(label, row, 0)
        grid_layout.addWidget(indicator, row, 1, Qt.AlignRight)
        grid_layout.addWidget(conn_btn, row, 2)
        grid_layout.addWidget(disconn_btn, row, 3)
        return indicator, (conn_btn, disconn_btn)

    def _setup_connections(self):
        # 设备连接
        self.cam_btns[0].clicked.connect(self.connect_camera_signal.emit)
        self.cam_btns[1].clicked.connect(self.disconnect_camera_signal.emit)
        self.arm_btns[0].clicked.connect(self.connect_arm_signal.emit)
        self.arm_btns[1].clicked.connect(self.disconnect_arm_signal.emit)
        self.gripper_btns[0].clicked.connect(self.connect_gripper_signal.emit)
        self.gripper_btns[1].clicked.connect(self.disconnect_gripper_signal.emit)
        # 目标识别和抓取预测开关
        self.enable_detection_cb.toggled.connect(self.detection_toggle_signal.emit)
        self.enable_grasp_cb.toggled.connect(self.grasp_toggle_signal.emit)
        # 任务控制
        self.mode_group.buttonClicked.connect(self.update_mode_ui)
        self.send_inst_button.clicked.connect(lambda: self.instruction_signal.emit(self.instruction_text.toPlainText()))
        self.strategy_combo.currentTextChanged.connect(self.strategy_signal.emit)
        # 执行控制
        self.execute_button.clicked.connect(self.execute_grasp_signal.emit)
        self.stop_button.clicked.connect(self.stop_signal.emit)

    @pyqtSlot()
    def update_mode_ui(self):
        is_inst_mode = self.inst_mode_radio.isChecked()
        self.strategy_combo.setEnabled(is_inst_mode)
        self.instruction_text.setVisible(is_inst_mode)
        self.send_inst_button.setVisible(is_inst_mode)
        self.auto_mode_button.setVisible(not is_inst_mode)

    @pyqtSlot(str, bool)
    def update_device_status(self, device, is_connected):
        status_indicator, (conn_btn, disconn_btn) = {
            "cam": (self.cam_status, self.cam_btns),
            "arm": (self.arm_status, self.arm_btns),
            "gripper": (self.gripper_status, self.gripper_btns)
        }.get(device, (None, (None, None)))

        if status_indicator:
            color = "#A3BE8C" if is_connected else "#BF616A"
            status_indicator.setStyleSheet(f"background-color: {color}")
            conn_btn.setEnabled(not is_connected)
            disconn_btn.setEnabled(is_connected)

        if device == "cam":
            self.enable_detection_cb.setEnabled(is_connected)
            self.enable_grasp_cb.setEnabled(is_connected)
            if not is_connected:
                self.enable_detection_cb.setChecked(False)
                self.enable_grasp_cb.setChecked(False)


class VisionLogPanel(QFrame):
    """中间视觉与日志面板"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(self)
        self.main_video_label = QLabel("主视觉区")
        self.main_video_label.setAlignment(Qt.AlignCenter)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        layout.addWidget(self.main_video_label, 3)
        layout.addWidget(self.log_box, 1)


class AnalysisPanel(QFrame):
    """右侧分析面板"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(self)

        def create_display(name):
            label = QLabel(f"<b>{name}</b>")
            label.setAlignment(Qt.AlignCenter)
            display = QLabel()
            display.setAlignment(Qt.AlignCenter)
            display.setFrameShape(QFrame.Box)
            display.setStyleSheet("background-color: #2E3440")
            display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            return label, display

        self.roi_title, self.roi_display = create_display("目标物体 (ROI)")
        self.q_title, self.q_display = create_display("抓取质量图 (Q)")
        self.ang_title, self.ang_display = create_display("抓取角度图 (Angle)")
        self.width_title, self.width_display = create_display("抓取宽度图 (Width)")

        layout.addWidget(self.roi_title)
        layout.addWidget(self.roi_display, 1)
        layout.addWidget(self.q_title)
        layout.addWidget(self.q_display, 1)
        layout.addWidget(self.ang_title)
        layout.addWidget(self.ang_display, 1)
        layout.addWidget(self.width_title)
        layout.addWidget(self.width_display, 1)
