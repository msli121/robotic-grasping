# -*- coding: utf-8 -*-
# @Time       : 2025/9/6 17:59
# @File       : ui_logger.py.py
# @Description: UI彩色日志生成器
import time


class UILogger:
    """生成用于UI显示的带颜色的HTML日志"""
    COLOR_INFO = "#D8DEE9"  # 浅灰色 (普通信息)
    COLOR_SUCCESS = "#A3BE8C"  # 绿色 (成功)
    COLOR_WARNING = "#EBCB8B"  # 黄色 (等待/警告)
    COLOR_ERROR = "#BF616A"  # 红色 (错误)
    COLOR_SYSTEM = "#88C0D0"  # 蓝色 (系统/任务指令)

    @staticmethod
    def now_str():
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    @staticmethod
    def _create_log_html(prefix, message, color):
        # 使用HTML <p> 标签来格式化，margin:0; padding:0; 使日志更紧凑
        return f'<p style="color:{color}; margin:1px; padding:0;">' \
               f'<b>{prefix}</b> {message}</p>'

    @staticmethod
    def info(message):
        return UILogger._create_log_html(f"[{UILogger.now_str()}]", message, UILogger.COLOR_INFO)

    @staticmethod
    def success(message):
        return UILogger._create_log_html(f"[{UILogger.now_str()}]", message, UILogger.COLOR_SUCCESS)

    @staticmethod
    def warning(message):
        return UILogger._create_log_html(f"[{UILogger.now_str()}]", message, UILogger.COLOR_WARNING)

    @staticmethod
    def error(message):
        return UILogger._create_log_html(f"[{UILogger.now_str()}]", message, UILogger.COLOR_ERROR)

    @staticmethod
    def system(message):
        return UILogger._create_log_html("&gt;&gt;", message, UILogger.COLOR_SYSTEM)
