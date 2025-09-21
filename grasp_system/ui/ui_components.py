from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt
from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
                             QTextEdit, QComboBox, QRadioButton, QButtonGroup,
                             QFrame, QGridLayout, QSizePolicy, QCheckBox)


class ControlPanel(QFrame):
    """
    左侧控制面板, 只负责UI元素的创建和布局, 通过信号向外发送用户操作。
    将模式切换升级为功能更明确的“识别模式”切换。
    """
    # --- 定义发出的信号 ---
    # 设备控制
    connect_camera_signal = pyqtSignal()
    disconnect_camera_signal = pyqtSignal()
    connect_arm_signal = pyqtSignal()
    disconnect_arm_signal = pyqtSignal()
    connect_gripper_signal = pyqtSignal()
    disconnect_gripper_signal = pyqtSignal()

    # 功能开关
    detection_toggle_signal = pyqtSignal(bool)
    grasp_toggle_signal = pyqtSignal(bool)

    # 任务控制
    mode_changed_signal = pyqtSignal(str)  # 发射模式名称: "open_vocab" 或 "closed_set"
    instruction_signal = pyqtSignal(str)
    clear_instruction_signal = pyqtSignal() # 用于通知后端清空指令
    strategy_signal = pyqtSignal(str)

    # 执行控制
    execute_grasp_signal = pyqtSignal()
    stop_signal = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self._init_ui()
        self._setup_connections()

    def _init_ui(self):
        """初始化所有UI元素并进行布局。"""
        layout = QVBoxLayout(self)

        # 创建并添加各个功能区域的框架
        device_frame = self._create_device_frame()
        task_frame = self._create_task_frame()
        exec_frame = self._create_exec_frame()

        layout.addWidget(device_frame)
        layout.addStretch()
        layout.addWidget(task_frame)
        layout.addStretch()
        layout.addWidget(exec_frame)
        layout.addStretch(4)

    def _create_device_frame(self):
        """创建设备连接区域。"""
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
        """创建任务控制区域。"""
        frame = QFrame()
        layout = QVBoxLayout(frame)
        layout.addWidget(QLabel("<b>任务控制</b>"))

        # 功能开关
        layout.addWidget(QLabel("功能开关:"))
        self.enable_detection_cb = QCheckBox("开启目标识别")
        self.enable_grasp_cb = QCheckBox("开启抓取预测")
        self.enable_detection_cb.setEnabled(False)
        self.enable_grasp_cb.setEnabled(False)
        switch_hbox = QHBoxLayout()
        switch_hbox.addWidget(self.enable_detection_cb)
        switch_hbox.addWidget(self.enable_grasp_cb)
        layout.addLayout(switch_hbox)

        # --- 核心改动: 替换为“识别模式”切换 ---
        layout.addWidget(QLabel("识别模式:"))
        self.open_vocab_radio = QRadioButton("开放词汇 (灵活)")
        self.closed_set_radio = QRadioButton("闭集专家 (高精度)")
        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.open_vocab_radio)
        self.mode_group.addButton(self.closed_set_radio)
        self.closed_set_radio.setChecked(True)  # 默认使用高精度的闭集模式

        mode_hbox = QHBoxLayout()
        mode_hbox.addWidget(self.open_vocab_radio)
        mode_hbox.addWidget(self.closed_set_radio)
        layout.addLayout(mode_hbox)

        layout.addWidget(QLabel("目标选择策略:"))
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["置信度最高", "抓取最近的", "从左到右"])
        layout.addWidget(self.strategy_combo)

        layout.addWidget(QLabel("文本指令:"))
        self.instruction_text = QTextEdit()
        self.instruction_text.setPlaceholderText("在此输入指令，例如：抓取红色打火机...")
        layout.addWidget(self.instruction_text)

        # 创建一个水平布局来放置两个按钮
        button_hbox = QHBoxLayout()
        # 创建“清空指令”按钮
        self.clear_inst_button = QPushButton("清空指令")
        # 创建“发送指令”按钮
        self.send_inst_button = QPushButton("发送指令")
        # 将两个按钮添加到水平布局中
        button_hbox.addWidget(self.clear_inst_button)
        button_hbox.addWidget(self.send_inst_button)
        # 将这个水平布局添加到主垂直布局中
        layout.addLayout(button_hbox)

        return frame

    def _create_exec_frame(self):
        """创建执行控制区域。"""
        frame = QFrame()
        layout = QVBoxLayout(frame)
        self.execute_button = QPushButton("执行抓取")
        self.execute_button.setEnabled(False)
        self.stop_button = QPushButton("强制停止")
        layout.addWidget(self.execute_button)
        layout.addWidget(self.stop_button)
        return frame

    def _create_device_row(self, grid_layout, row, name):
        """辅助函数，用于创建一行设备控制UI。"""
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
        """连接所有UI元素的信号到此类定义的信号上。"""
        # 设备连接
        self.cam_btns[0].clicked.connect(self.connect_camera_signal.emit)
        self.cam_btns[1].clicked.connect(self.disconnect_camera_signal.emit)
        self.arm_btns[0].clicked.connect(self.connect_arm_signal.emit)
        self.arm_btns[1].clicked.connect(self.disconnect_arm_signal.emit)
        self.gripper_btns[0].clicked.connect(self.connect_gripper_signal.emit)
        self.gripper_btns[1].clicked.connect(self.disconnect_gripper_signal.emit)

        # 功能开关
        self.enable_detection_cb.toggled.connect(self.detection_toggle_signal.emit)
        self.enable_grasp_cb.toggled.connect(self.grasp_toggle_signal.emit)

        # 任务控制
        self.mode_group.buttonClicked.connect(self._on_mode_changed)  # 连接到内部槽
        self.clear_inst_button.clicked.connect(self._on_clear_instruction_clicked)
        self.send_inst_button.clicked.connect(lambda: self.instruction_signal.emit(self.instruction_text.toPlainText()))
        self.strategy_combo.currentTextChanged.connect(self.strategy_signal.emit)

        # 执行控制
        self.execute_button.clicked.connect(self.execute_grasp_signal.emit)
        self.stop_button.clicked.connect(self.stop_signal.emit)

    @pyqtSlot()
    def _on_clear_instruction_clicked(self):
        """
        “清空指令”按钮被点击时触发的槽函数。
        负责执行UI清理并通知后端。
        """
        # 1. 清空文本输入框 (UI操作)
        self.instruction_text.clear()
        # 2. 发射信号，通知后端状态已清空
        self.clear_instruction_signal.emit()
        # 3. (未来可扩展) 在这里添加其他您需要的UI功能
        # 例如，将焦点重新设置回文本框
        self.instruction_text.setFocus()

    @pyqtSlot()
    def _on_mode_changed(self):
        """内部槽函数，用于发射带有模式名称字符串的信号。"""
        if self.open_vocab_radio.isChecked():
            self.mode_changed_signal.emit("open_vocab")
        else:
            self.mode_changed_signal.emit("closed_set")

    @pyqtSlot(str, bool)
    def update_device_status(self, device, is_connected):
        """槽函数，用于更新设备状态UI。"""
        status_indicator, (conn_btn, disconn_btn) = {
            "cam": (self.cam_status, self.cam_btns),
            "arm": (self.arm_status, self.arm_btns),
            "gripper": (self.gripper_status, self.gripper_btns)
        }.get(device, (None, (None, None)))

        if status_indicator:
            color = "#A3BE8C" if is_connected else "#BF616A"  # 绿色/红色
            status_indicator.setStyleSheet(f"background-color: {color}")
            conn_btn.setEnabled(not is_connected)
            disconn_btn.setEnabled(is_connected)

        # 联动逻辑: 只有相机连接后，功能开关才可用
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
        self.main_video_label = QLabel("无视频流～～")
        self.main_video_label.setAlignment(Qt.AlignCenter)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        # TextSelectableByMouse: 允许用鼠标选择文本
        # TextSelectableByKeyboard: 允许用键盘 (e.g., Ctrl+A) 选择文本
        self.log_box.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
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
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
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
