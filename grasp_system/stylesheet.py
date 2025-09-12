# -*- coding: utf-8 -*-
# @Time       : 2025/9/6 17:58
# @File       : stylesheet.py.py
# @Description: UI样式表

STYLE_SHEET = """
    /* 主窗口和框架 */
    QMainWindow, QWidget {
        background-color: #2E3440; /* 深灰蓝背景 */
    }

    QFrame {
        background-color: #3B4252; /* 稍亮的框架背景 */
        border-radius: 8px;
    }

    /* 标签 */
    QLabel {
        color: #ECEFF4; /* 浅灰白色字体 */
        font-size: 14px;
        font-family: Arial, Helvetica, sans-serif;
    }

    /* 按钮 */
    QPushButton {
        background-color: #4C566A; /* 灰色按钮 */
        color: #ECEFF4;
        border: none;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    QPushButton:hover {
        background-color: #5E81AC; /* 悬停时变蓝 */
    }
    QPushButton:pressed {
        background-color: #81A1C1; /* 按下时变浅蓝 */
    }
    QPushButton:disabled {
        background-color: #3B4252; /* 禁用时变暗 */
        color: #4C566A;
    }

    /* 文本输入框 */
    QLineEdit, QTextEdit {
        background-color: #434C5E;
        color: #D8DEE9;
        border: 1px solid #4C566A;
        border-radius: 5px;
        padding: 5px;
        font-size: 14px;
    }

    /* 下拉菜单 */
    QComboBox {
        background-color: #434C5E;
        color: #D8DEE9;
        border: 1px solid #4C566A;
        border-radius: 5px;
        padding: 5px;
    }
    QComboBox::drop-down {
        border: none;
    }

    /* 单选框 */
    QRadioButton {
        color: #ECEFF4; /* 文本颜色设为浅灰白色 */
        font-size: 14px;
        spacing: 5px; /* 图标和文本之间的间距 */
    }
    QRadioButton::indicator {
        width: 15px;
        height: 15px;
    }
    QRadioButton::indicator::unchecked {
        background-color: #4C566A; /* 未选中时的背景色 */
        border: 1px solid #D8DEE9;
        border-radius: 7px;
    }
    QRadioButton::indicator::checked {
        background-color: #88C0D0; /* 选中时变蓝色 */
        border: 1px solid #D8DEE9;
        border-radius: 7px;
    }

    /* 专门用于状态指示灯的QLabel */
    QLabel#status_indicator {
        max-height: 16px;
        max-width: 16px;
        border-radius: 8px; /* 圆形 */
    }
    
    /* 复选框 */
    QCheckBox {
        color: #ECEFF4; /* 文本颜色设为浅灰白色 */
        font-size: 14px;
        spacing: 5px; /* 图标和文本之间的间距 */
    }
    QCheckBox::indicator {
        width: 15px;
        height: 15px;
    }
    QCheckBox::indicator::unchecked {
        background-color: #4C566A; /* 未选中时的背景色 */
        border: 1px solid #D8DEE9;
        border-radius: 3px; /* 方形带圆角 */
    }
    QCheckBox::indicator::checked {
        background-color: #88C0D0; /* 选中时变蓝色 */
        border: 1px solid #D8DEE9;
        border-radius: 3px; /* 方形带圆角 */
        /* 可选: 添加一个勾选的图像来美化 */
        /* image: url(./icons/check-mark.png); */
    }
"""