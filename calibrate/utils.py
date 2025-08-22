# -*- coding: utf-8 -*-
# @File       : pre_process_data.py
# @Description: 处理标定板图像和机器人位姿数据，用于完成后续手眼标定功能
# @Author     : lms
# @Date       : 2025/8/16 21:06

import logging
import math
import os
import re
import shutil
from scipy.spatial.transform import Rotation as R

import cv2
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
np.set_printoptions(precision=8, suppress=True)


def euler_angles_to_rotation_matrix(rx, ry, rz):
    # 计算旋转矩阵
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
    # zyx
    R = Rz @ Ry @ Rx
    # xyz
    # R = Rx @ Ry @ Rz
    return R


def euler_to_rotation_matrix_scipy(rx, ry, rz, order='zyx', degrees=False):
    """
    【推荐】使用scipy库将欧拉角转换为旋转矩阵，健壮且高效。

    参数:
    rx, ry, rz (float): 分别绕X, Y, Z轴的旋转角度。
    order (str): 欧拉角的旋转顺序。对于机器人，这通常是'zyx'（内旋）。
                 Scipy支持所有12种序列: 'xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx'
                 以及 'xyx', 'xzx', 'yxy', 'yzy', 'zxz', 'zyz'。
    degrees (bool): 如果为True，则输入角度单位为度；否则为弧度。

    返回:
    np.ndarray: 3x3的旋转矩阵。
    """
    # 注意：scipy的from_euler函数需要一个与order字符串顺序匹配的角度列表。
    # 例如，如果order是'zyx'，角度列表必须是[rz, ry, rx]。
    angle_map = {'x': rx, 'y': ry, 'z': rz}
    angles_in_order = [angle_map[axis] for axis in order]

    rotation_obj = R.from_euler(order, angles_in_order, degrees=degrees)
    return rotation_obj.as_matrix()


def pose_to_homogeneous_matrix(pose, order='zyx'):
    x, y, z, rx, ry, rz = pose
    R = euler_to_rotation_matrix_scipy(rx, ry, rz, order=order)
    t = np.array([x, y, z]).reshape(3, 1)
    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = t[:, 0]
    return H


def robot_pose_to_homogeneous_matrix(robot_pose, order='zyx'):
    """
    将机器人位姿转换为齐次变换矩阵。

    参数:
    robot_pose (list): 机器人位姿，包含6个元素 [x, y, z, rx, ry, rz]，
                       分别为位置和欧拉角（位置单位为毫米 默认单位为度）。
    order (str): 欧拉角的旋转顺序，默认'zyx'。
    degrees (bool): 如果为True，则输入角度单位为度；否则为弧度。

    返回:
    np.ndarray: 4x4的齐次变换矩阵。
    """
    # 判断输入是否为列表 或者 np的一维数组
    if not isinstance(robot_pose, (list, np.ndarray)):
        raise ValueError("输入必须是一个列表或 numpy 数组")
    if len(robot_pose) != 6:
        raise ValueError("输入列表必须包含6个元素")
    # 前3个毫米转米
    robot_pose[:3] = np.array(robot_pose[:3]) / 1000
    # 后3个度转弧度
    robot_pose[3:] = np.deg2rad(robot_pose[3:])
    # 转换为齐次变换矩阵
    return pose_to_homogeneous_matrix(robot_pose, order=order)


def normalize_corner_order(corners, checkerboard_size):
    """
    统一OpenCV棋盘格角点的检测顺序，确保总是从左上角开始，逐行扫描。

    参数:
    corners (np.ndarray): cv2.findChessboardCorners 或 cornerSubPix 返回的角点数组。
                          形状应为 (rows*cols, 1, 2)。
    checkerboard_size (tuple): 棋盘格的内角点数量，格式为 (cols, rows)，例如 (8, 8)。

    返回:
    np.ndarray: 顺序被归一化后的角点数组。
    """
    # 将角点数组展平以便于计算，形状变为 (N, 2)
    corners_flat = np.squeeze(corners)

    # 1. 利用几何特性找到四个最外侧的角点
    # x+y 最小的是左上角
    sum_xy = corners_flat.sum(axis=1)
    top_left_idx = np.argmin(sum_xy)

    # x+y 最大的是右下角
    bottom_right_idx = np.argmax(sum_xy)

    # x-y 最大的是右上角
    diff_xy = np.diff(corners_flat, axis=1).flatten()
    top_right_idx = np.argmax(diff_xy)

    # y-x 最大的是左下角 (等价于 x-y 最小)
    bottom_left_idx = np.argmin(diff_xy)

    # 2. 判断检测到的第一个角点 corner[0] 是哪个物理角点
    # 我们用索引来判断，因为浮点数直接比较可能不稳定
    first_corner_idx = 0

    # 获取检测顺序的起始角点
    # 注意：为了处理可能的浮点误差，我们比较索引而不是坐标值
    if first_corner_idx == top_left_idx:
        # 理想情况：顺序已经是正确的 (左上角 -> 右下角)
        # logger.debug("角点顺序正确 (TL-BR)")
        return corners

    elif first_corner_idx == top_right_idx:
        # 情况2：顺序为 右上角 -> 左下角
        # 需要对每一行进行水平翻转
        # logger.debug("角点顺序修正 (TR-BL -> TL-BR)")
        rows, cols = checkerboard_size[1], checkerboard_size[0]
        # 保持原始数据类型和形状
        corrected_corners = corners.reshape(rows, cols, 1, 2)
        corrected_corners = corrected_corners[:, ::-1, :, :]  # 对列（cols）进行翻转
        return corrected_corners.reshape(-1, 1, 2)

    elif first_corner_idx == bottom_left_idx:
        # 情况3：顺序为 左下角 -> 右上角
        # 需要对整个数组进行垂直翻转
        # logger.debug("角点顺序修正 (BL-TR -> TL-BR)")
        return corners[::-1]

    elif first_corner_idx == bottom_right_idx:
        # 情况4：顺序为 右下角 -> 左上角
        # 需要进行水平和垂直双重翻转
        # logger.debug("角点顺序修正 (BR-TL -> TL-BR)")
        corrected_corners = corners[::-1]  # 先垂直翻转
        rows, cols = checkerboard_size[1], checkerboard_size[0]
        corrected_corners = corrected_corners.reshape(rows, cols, 1, 2)
        corrected_corners = corrected_corners[:, ::-1, :, :]
        return corrected_corners.reshape(-1, 1, 2)
    else:
        # 这是一个异常情况，第一个角点不是四个角之一，可能检测有误
        logger.warning("无法确定角点检测顺序，可能检测结果有误。")
        return corners  # 返回原始值，让后续流程处理


def process_checkerboard_and_pose_data(source_dir):
    """
    处理标定板图像和机器人位姿数据，完成指定功能

    参数:
        source_dir: 包含x_rgb.png和x_pos.txt文件的源文件夹路径
    """
    checkerboard_dir = "./checkerboard_images"
    robot_pose_file = "./robot_tcp_pose.txt"

    # 1. 删除./checkerboard_images下所有文件
    os.makedirs(checkerboard_dir, exist_ok=True)  # 确保目录存在
    for filename in os.listdir(checkerboard_dir):
        file_path = os.path.join(checkerboard_dir, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            raise RuntimeError(f"删除文件 {file_path} 失败: {str(e)}")

    # 2. 获取并验证源文件夹中的x_rgb.png和对应x_pos.txt
    if not os.path.isdir(source_dir):
        raise NotADirectoryError(f"源文件夹 {source_dir} 不存在或不是目录")

    # 匹配x_rgb.png格式的文件并提取编号x
    rgb_files = []
    for filename in os.listdir(source_dir):
        if filename.endswith("_rgb.png"):
            x = filename.split("_")[0]
            rgb_files.append((x, filename))

    # 按编号升序排序
    rgb_files.sort(key=lambda item: item[0])
    if not rgb_files:
        raise ValueError(f"源文件夹 {source_dir} 中未找到符合格式的x_rgb.png文件")

    # 验证对应x_pos.txt是否存在
    valid_xs = []
    for x, rgb_filename in rgb_files:
        pos_filename = f"{x}_pos.txt"
        pos_path = os.path.join(source_dir, pos_filename)
        if not os.path.exists(pos_path):
            raise FileNotFoundError(f"缺少与 {rgb_filename} 对应的位置文件: {pos_filename}")
        valid_xs.append(x)

    # 3. 复制并重新命名x_rgb.png到checkerboard_images
    for idx, (x, rgb_filename) in enumerate(rgb_files, start=1):
        src_path = os.path.join(source_dir, rgb_filename)
        dst_path = os.path.join(checkerboard_dir, f"{idx}.png")
        try:
            shutil.copy(src_path, dst_path)
        except Exception as e:
            raise RuntimeError(f"移动文件 {src_path} 失败: {str(e)}")

    # 4. 处理机器人位姿文件
    # 清空现有内容
    with open(robot_pose_file, 'w', encoding='utf-8') as f:
        pass

    # 解析并写入转换后的数据
    with open(robot_pose_file, 'a', encoding='utf-8') as pose_f:
        for x in valid_xs:
            pos_path = os.path.join(source_dir, f"{x}_pos.txt")
            try:
                with open(pos_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 提取data行数据
                data_match = re.search(r'data: \[(.*?)\]', content)
                if not data_match:
                    raise ValueError(f"文件 {pos_path} 中未找到有效的data数据")

                data_str = data_match.group(1)
                data_list = data_str.split(', ')
                if len(data_list) < 6:
                    raise ValueError(f"文件 {pos_path} 中data数据不足6项")

                # 转换单位
                try:
                    x_mm, y_mm, z_mm = map(float, data_list[:3])
                    rx_deg, ry_deg, rz_deg = map(float, data_list[3:6])
                except ValueError:
                    raise ValueError(f"文件 {pos_path} 中data数据格式错误")

                # 毫米转米，度转弧度
                x_m = x_mm / 1000.0
                y_m = y_mm / 1000.0
                z_m = z_mm / 1000.0
                rx_rad = math.radians(rx_deg)
                ry_rad = math.radians(ry_deg)
                rz_rad = math.radians(rz_deg)

                # 写入文件（保留6位小数）
                pose_line = f"{x_m:.6f} {y_m:.6f} {z_m:.6f} {rx_rad:.6f} {ry_rad:.6f} {rz_rad:.6f}\n"
                pose_f.write(pose_line)

            except Exception as e:
                raise RuntimeError(f"处理位置文件 {pos_path} 失败: {str(e)}")

    print(f"处理完成！\n"
          f"共处理 {len(rgb_files)} 组数据\n"
          f"图像已保存至: {checkerboard_dir}\n"
          f"位姿数据已保存至: {robot_pose_file}")


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
    # print(euler_to_rotation_matrix_scipy(0, 0, 0))
    # # 处理captures目录下的文件（可根据实际情况修改）
    # source_dir = "./captures"
    # # 处理拍摄的照片文件
    # process_checkerboard_and_pose_data(source_dir)

    # 标定板坐标系到法兰盘坐标系的变换矩阵
    robot_pose = [-19.3485, -79.2081, 199.392, -90, 0.0, 90]
    # pose = [-0.01935, -0.0, 0.1994, - np.pi / 2, 0, np.pi / 2]
    M_flange_board = robot_pose_to_homogeneous_matrix(robot_pose, order='xyz')
    print("标定板坐标系到法兰盘坐标系的变换矩阵")
    print(M_flange_board)

    # test_board_pose = [0.018, 0.018, 0.0, 1]
    test_board_pose = [0.0, 0.0, 0.0, 1]
    test_board_pose = np.asarray(test_board_pose).reshape((4, 1))
    robot_pose = M_flange_board @ test_board_pose
    print(f"标定板 => 法兰盘 {test_board_pose.flatten()[:3]} => {robot_pose.flatten()[:3]}")

    # 法兰盘坐标系到世界坐标系的变换矩阵
    flange_pose = [208.18, 71.42, 245.75, -152.60, -67.49, -38.95]
    M_base_flange = robot_pose_to_homogeneous_matrix(flange_pose, order='xyz')
    print("法兰盘坐标系到世界坐标系的变换矩阵")
    print(M_base_flange)

    M_base_board = M_base_flange @ M_flange_board
    print("标定板坐标系到机械臂基坐标系的变换矩阵")
    print(M_base_board)

    # 标定板坐标系到机械臂基坐标系的变换矩阵
    robot_pos = M_base_flange @ M_flange_board @ test_board_pose
    # [0 0 0] => [0.40530398 0.12049056 0.17405847]
    print(f"标定板 => 机械臂 {test_board_pose.flatten()[:3]} => {robot_pos.flatten()[:3]}")

    # captures_dir = "./captures"
    # # 确保目录存在
    # if not os.path.exists(captures_dir):
    #     logger.warning(f"目录 {captures_dir} 不存在，无法处理图像")
    # else:
    #     # 遍历目录下所有以_rgb.png结尾的文件
    #     for filename in os.listdir(captures_dir):
    #         if filename.endswith("_rgb.png"):
    #             rgb_path = os.path.join(captures_dir, filename)
    #             detect_and_save_corners(rgb_path, check_direction=False)
