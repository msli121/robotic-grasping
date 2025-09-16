import logging
import os
import sys
from colorlog import ColoredFormatter


def setup_logging():
    """
    配置项目的根记录器 (Root Logger)。
    一旦配置完成，所有子模块通过 logging.getLogger(__name__) 创建的 logger
    都会自动继承这里的配置。
    """
    # --- 1. 创建 logs 文件夹 ---
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 清除所有现有的 handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        handler.close()

    # --- 2. 获取根 Logger ---
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # --- 3. 配置控制台 Handler (彩色) ---
    console_log_format = '%(log_color)s%(asctime)s %(levelname)-8s: %(message)s (%(filename)s:%(lineno)d)'
    console_formatter = ColoredFormatter(
        console_log_format,
        datefmt='%Y-%m-%d %H:%M:%S',
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        }
    )

    # ==================== 核心修正点 ====================
    # 明确指定 StreamHandler 使用 sys.stdout，避免被 IDE 强制染成红色
    console_handler = logging.StreamHandler(sys.stdout)
    # ====================================================

    console_handler.setFormatter(console_formatter)

    # --- 4. 配置文件 Handler (非彩色) ---
    file_log_format = '%(asctime)s %(levelname)-8s: %(message)s (%(filename)s:%(lineno)d)'
    file_formatter = logging.Formatter(file_log_format, datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(os.path.join(log_dir, "app.log"), mode='a', encoding='utf-8')
    file_handler.setFormatter(file_formatter)

    # --- 5. 将 Handlers 添加到根 Logger ---
    if not root_logger.handlers:
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)

    # --- 6. (可选) 关闭第三方库的冗余日志 ---
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)