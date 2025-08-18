# -*- coding: utf-8 -*-
# @Time       : 2025/8/16 20:27
# @File       : draw_corner.py
# @Description: 检测标定板角点并保存带角点标记的图像
import logging
import os

import cv2

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def detect_and_save_corners(rgb_image_path, check_direction=True):
    """
    检测RGB图像中标定板的角点并保存带角点标记的图像
    如果已存在角点图像则跳过处理

    参数:
        rgb_image_path: RGB图像的路径
        check_direction: 是否检查角点方向，默认True
    """
    # 生成输出文件名
    base_name = os.path.splitext(rgb_image_path)[0]
    output_path = f"{base_name}_corner.png"

    # 检查是否已存在角点图像，存在则跳过
    if os.path.exists(output_path):
        logger.info(f"角点图像已存在: {output_path}，跳过处理")
        return True

    # 读取图像
    image = cv2.imread(rgb_image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {rgb_image_path}")

    # 标定板参数
    checkerboard_size = (8, 8)  # 标定板内角点数量 (宽, 高)
    refine_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 转换为灰度图
    gray_data = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 寻找标定板角点
    checkerboard_found, corners = cv2.findChessboardCorners(
        gray_data,
        checkerboard_size,
        None,
        cv2.CALIB_CB_ADAPTIVE_THRESH
    )

    # 检查是否检测到角点
    if checkerboard_found:
        # 亚像素级角点精化
        corners_refined = cv2.cornerSubPix(
            gray_data,
            corners,
            (11, 11),  # 搜索窗口大小
            (-1, -1),  # 死区大小
            refine_criteria
        )

        # 检查角点方向
        if check_direction and abs(corners[0][0][1] - corners[1][0][1]) > 10:
            logger.warning(f"图像 {rgb_image_path} 角点识别方向与实际方向不一致，跳过处理")
            return False

        # 在原图上绘制角点
        image_with_corners = image.copy()
        cv2.drawChessboardCorners(
            image_with_corners,
            checkerboard_size,
            corners_refined,
            checkerboard_found
        )

        # 保存带角点的图像
        cv2.imwrite(output_path, image_with_corners)
        logger.info(f"已检测到角点并保存至: {output_path}")
        return True
    else:
        logger.info(f"在图像 {rgb_image_path} 中未检测到标定板角点")
        return False


# 使用示例
if __name__ == "__main__":
    captures_dir = "./captures"
    # 确保目录存在
    if not os.path.exists(captures_dir):
        logger.warning(f"目录 {captures_dir} 不存在，无法处理图像")
    else:
        # 遍历目录下所有以_rgb.png结尾的文件
        for filename in os.listdir(captures_dir):
            if filename.endswith("_rgb.png"):
                rgb_path = os.path.join(captures_dir, filename)
                detect_and_save_corners(rgb_path, check_direction=False)
