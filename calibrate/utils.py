# -*- coding: utf-8 -*-
# @File       : pre_process_data.py
# @Description: 处理标定板图像和机器人位姿数据，用于完成后续手眼标定功能
# @Author     : lms
# @Date       : 2025/8/16 21:06
import ast
import glob
import logging
import math
import os
import re
import shutil
from time import process_time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
np.set_printoptions(precision=8, suppress=True)


def inverse_transformation_matrix(T):
    """
    求 4x4 齐次变换矩阵的逆
    """
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError("输入必须是 4x4 齐次矩阵")

    R = T[:3, :3]
    t = T[:3, 3]

    # 检查R是否近似正交
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-8):
        raise ValueError("旋转部分不是正交矩阵，输入可能不是合法SE(3)变换")

    R_inv = R.T
    t_inv = -R_inv @ t

    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv


def euler_angles_to_rotation_matrix(rx, ry, rz, order='ZYX'):
    """
    将欧拉角转换为旋转矩阵 (使用scipy库实现)

    根据手册，默认的旋转类型为内旋(Intrinsic)，顺序为Z-Y-X，因此默认 order='ZYX'。

    参数:
    rx, ry, rz (float): 绕X, Y, Z轴的欧拉角，单位必须是【弧度】(radians)。
    order (str): 旋转顺序和类型。
                 - 大写字母 (e.g., 'ZYX', 'XYZ'): 代表内旋 (Intrinsic)，每一次旋转都是围绕物体自身的、已经发生过旋转的新坐标系的轴进行的
                 - 小写字母 (e.g., 'zyx', 'xyz'): 代表外旋 (Extrinsic)，所有的旋转都是围绕固定的基座坐标系的轴进行的

    返回:
    np.ndarray: 3x3的旋转矩阵。
    """
    # R.from_euler 需要一个角度列表 [rx, ry, rz]
    # 注意！scipy的顺序是根据 order 字符串来的。
    # 如果 order 是 'ZYX', 那么它期望的角度列表是 [angle_z, angle_y, angle_x]
    angles_map = {'x': rx, 'y': ry, 'z': rz}

    # 根据order字符串的顺序重新排列角度列表
    # 例如 order='ZYX' -> [rz, ry, rx]
    # 这里使用 lower() 是因为 'z' 和 'Z' 都对应 rz
    ordered_angles = [angles_map[axis] for axis in order.lower()]

    # 使用scipy进行转换
    # 关键点:
    # 1. 传入正确的旋转顺序和类型 (order)
    # 2. 传入与 order 对应顺序的角度列表
    # 3. 指定输入角度的单位是弧度 (radians)，通过设置 degrees=False
    rotation_matrix = R.from_euler(order, ordered_angles, degrees=False).as_matrix()

    return rotation_matrix


def pose_to_homogeneous_matrix(pose, order='ZYX'):
    """
    将位姿转换为齐次变换矩阵。

    参数:
    pose (list): 位姿，包含6个元素 [x, y, z, rx, ry, rz]，
                 分别为位置和欧拉角（位置单位为毫米 默认单位为度）。
    order (str): 欧拉角的顺序，默认是 'xyz'。

    返回:
    np.ndarray: 4x4的齐次变换矩阵。
    """
    if pose is None or len(pose) < 6 or len(pose) > 7:
        raise ValueError('pose is invalid')
    if len(pose) == 7:
        pose = pose[:6]
    x, y, z, rx, ry, rz = pose
    R = euler_angles_to_rotation_matrix(rx, ry, rz, order=order)
    t = np.array([x, y, z]).reshape(3, 1)
    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = t[:, 0]
    return H


def robot_pose_to_homogeneous_matrix(robot_pose, order='ZYX'):
    """
    将机器人位姿转换为齐次变换矩阵。

    参数:
    robot_pose (list): 机器人位姿，包含6个元素 [x, y, z, rx, ry, rz]，
                       分别为位置和欧拉角（位置单位为毫米 默认单位为度）。

    返回:
    np.ndarray: 4x4的齐次变换矩阵。
    """
    # 判断输入是否为列表 或者 np的一维数组
    if not isinstance(robot_pose, (list, np.ndarray)):
        raise ValueError("输入必须是一个列表或 numpy 数组")
    if len(robot_pose) == 7:
        robot_pose = robot_pose[:6]
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
    一个稳健的函数，用于统一OpenCV棋盘格角点的检测顺序。
    无论原始扫描方向如何，都确保输出为“左上角起始，逐行扫描”。

    参数:
    corners (np.ndarray): cv2.findChessboardCorners 或 cornerSubPix 返回的角点数组。
                          形状应为 (rows*cols, 1, 2)。
    checkerboard_size (tuple): 棋盘格的内角点数量，格式为 (cols, rows)，例如 (8, 8)。

    返回:
    np.ndarray: 顺序被归一化后的角点数组，形状为 (rows*cols, 1, 2)。
    """
    # 确保 checkerboard_size 是 (cols, rows)
    cols, rows = checkerboard_size

    if corners.shape[0] != rows * cols:
        logger.error(f"角点数量 ({corners.shape[0]}) 与棋盘格尺寸 ({rows * cols}) 不匹配。")
        return corners  # 无法处理，返回原值

    # 将角点数组展平以便于计算，形状变为 (N, 2)
    corners_flat = np.squeeze(corners)

    # 1. 利用几何特性找到四个最外侧的角点
    # 在图像坐标系中, y轴是向下的。
    # 左上角 (TL): x+y 最小
    sum_xy = corners_flat.sum(axis=1)
    top_left_idx = np.argmin(sum_xy)

    # 右下角 (BR): x+y 最大
    bottom_right_idx = np.argmax(sum_xy)

    # ========================= 【核心修正】 =========================
    # 我们需要计算 x-y 来区分右上角和左下角
    # 右上角 (TR): x最大, y最小 -> x-y 最大
    # 左下角 (BL): x最小, y最大 -> x-y 最小
    diff_x_minus_y = corners_flat[:, 0] - corners_flat[:, 1]
    top_right_idx = np.argmax(diff_x_minus_y)
    bottom_left_idx = np.argmin(diff_x_minus_y)
    # ===============================================================

    # 获取四个角点的坐标
    p_tl = corners_flat[top_left_idx]
    p_tr = corners_flat[top_right_idx]
    p_bl = corners_flat[bottom_left_idx]

    # 2. 定义棋盘格的坐标轴（不要求严格正交）
    # 水平方向向量 (从左上到右上)
    vec_horizontal = p_tr - p_tl
    # 垂直方向向量 (从左上到左下)
    vec_vertical = p_bl - p_tl

    # 归一化方向向量
    vec_horizontal_norm = vec_horizontal / np.linalg.norm(vec_horizontal)
    vec_vertical_norm = vec_vertical / np.linalg.norm(vec_vertical)

    # 3. 计算所有点相对于左上角点的投影
    vectors_from_tl = corners_flat - p_tl
    proj_horizontal = np.dot(vectors_from_tl, vec_horizontal_norm)
    proj_vertical = np.dot(vectors_from_tl, vec_vertical_norm)

    # 4. 根据投影进行排序
    # 计算平均行间距，用于确定每个点所属的行
    avg_dist_per_row = np.linalg.norm(p_bl - p_tl) / (rows - 1)

    # 根据垂直投影将点分组到不同的行
    # y_indices 代表每个点属于哪一行
    y_indices = np.round(proj_vertical / avg_dist_per_row).astype(int)

    # 将所有信息组合起来: (行索引, 水平投影, 原始坐标)
    combined = list(zip(y_indices, proj_horizontal, corners_flat))

    # 先按行索引排序，再按水平投影排序 (实现Z字形扫描)
    combined.sort(key=lambda k: (k[0], k[1]))

    # 提取排好序的角点坐标
    sorted_corners_flat = np.array([item[2] for item in combined])

    # 恢复到OpenCV期望的形状 (N, 1, 2)
    return sorted_corners_flat.reshape(-1, 1, 2)


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


def calculate_transform_error(T1, T2):
    """
    计算两个4x4齐次变换矩阵之间的平移和旋转误差。

    这个函数是验证标定精度的核心，用于比较两个理论上应该相等的位姿矩阵，
    例如 AX 和 XB。

    参数:
    T1 (np.ndarray): 第一个4x4齐次变换矩阵。
    T2 (np.ndarray): 第二个4x4齐次变换矩阵。

    返回:
    tuple: 一个包含两个元素的元组:
           - translation_error (float): 平移误差，单位与输入矩阵相同 (通常是米)。
           - rotation_error_deg (float): 旋转误差，单位是度 (degrees)。
    """
    # 1. 提取平移向量
    # 平移向量是矩阵最后一列的前三个元素
    t1 = T1[:3, 3]
    t2 = T2[:3, 3]

    # 2. 计算平移误差
    # 平移误差是两个平移向量之间的欧氏距离
    translation_error = np.linalg.norm(t1 - t2)

    # 3. 提取旋转矩阵
    # 旋转矩阵是左上角的3x3子矩阵
    R1 = T1[:3, :3]
    R2 = T2[:3, :3]

    # 4. 计算旋转误差
    # 误差旋转矩阵 R_diff 表示从 R2 旋转到 R1 需要的旋转
    # R_diff * R2 = R1  =>  R_diff = R1 * R2^T (因为R的逆是R的转置)
    R_diff = np.dot(R1, R2.T)

    # 从误差旋转矩阵中提取等效的旋转角度
    # 公式: angle = arccos((trace(R_diff) - 1) / 2)
    trace = np.trace(R_diff)

    # 由于浮点计算误差，trace的值可能略微超出 arccos 的定义域 [-1, 1]
    # 例如，计算出 1.0000001。必须将其裁剪到范围内，否则会引发数学错误。
    # trace 的范围是 [-1, 3]
    trace = np.clip(trace, -1.0, 3.0)

    angle_rad = np.arccos((trace - 1.0) / 2.0)

    # 将弧度转换为度
    rotation_error_deg = np.rad2deg(angle_rad)

    return translation_error, rotation_error_deg


def read_robot_poses(robot_post_file=None):
    """

    """
    if not robot_post_file or not os.path.exists(robot_post_file):
        raise ValueError("姿态文件不存在")
    poses = []

    with open(robot_post_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue  # 跳过空行

            try:
                # 使用ast.literal_eval安全解析列表字符串
                pose_list = ast.literal_eval(line)
                if not isinstance(pose_list, list):
                    raise ValueError("行内容不是有效的列表格式")
                poses.append(pose_list)
            except (SyntaxError, ValueError) as e:
                raise ValueError(f"文件{robot_post_file}第{line_num}行解析错误: {str(e)}") from e
    print(f"成功读取{len(poses)}个姿态数据")
    return poses


def test_process_position_files():
    # 定义文件路径
    captures_dir = r'D:\PycharmProjects\robotic-grasping\calibrate\captures'
    output_path = r'D:\PycharmProjects\robotic-grasping\calibrate\robot_origin_pose.txt'

    # 获取所有_pos.txt文件
    pos_files = glob.glob(os.path.join(captures_dir, '*_pos.txt'))
    if not pos_files:
        raise FileNotFoundError(f"在{captures_dir}目录下未找到任何_pos.txt文件")

    # 处理每个文件并写入结果
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for file_path in pos_files:
            with open(file_path, 'r', encoding='utf-8') as infile:
                content = infile.read()

            # 提取data行数据
            data_match = re.search(r'data: \[(.*?)\]', content)
            if not data_match:
                raise ValueError(f"文件{file_path}中未找到有效的data数据")

            data_str = data_match.group(1)
            data_list = data_str.split(', ')
            if len(data_list) < 6:
                raise ValueError(f"文件{file_path}中data数据不足6项")

            # 数据转换与四舍五入处理
            try:
                data_processed = [round(float(value.strip()), 6) for value in data_list]
            except ValueError as e:
                raise ValueError(f"文件{file_path}数据格式错误: {str(e)}") from e

            # 写入结果文件
            outfile.write(f"{data_processed}\n")

    print(f"成功处理{len(pos_files)}个文件，结果已保存至{output_path}")
    return output_path

# 使用示例
if __name__ == "__main__":
    output_path = test_process_position_files()
    robot_poses = read_robot_poses(output_path)
    print(robot_poses)
    #
    # # # 处理captures目录下的文件（可根据实际情况修改）
    # # source_dir = "./captures"
    # # # 处理拍摄的照片文件
    # # process_checkerboard_and_pose_data(source_dir)
    #
    # # 标定板坐标系到法兰盘坐标系的变换矩阵
    # # robot_pose = [-19.3485, -79.2081, 199.392, -90, 0.0, 90]
    # robot_pose = [-1.86159, -80.0191, 219.572, -90, 0, 90]
    # # pose = [-0.01935, -0.0, 0.1994, - np.pi / 2, 0, np.pi / 2]
    # M_flange_board = robot_pose_to_homogeneous_matrix(robot_pose, order='ZYX')
    # print("标定板坐标系到法兰盘坐标系的变换矩阵")
    # print(M_flange_board)
    #
    # # test_board_pose = [0.0, 0.018 * 11, 0.0, 1]
    # test_board_pose = [0.0, 0.0, 0.0, 1]
    # test_board_pose = np.asarray(test_board_pose).reshape((4, 1))
    # robot_pose = M_flange_board @ test_board_pose
    # print(f"标定板 => 法兰盘 {test_board_pose.flatten()[:3]} => {robot_pose.flatten()[:3]}")
    #
    # # 法兰盘坐标系到世界坐标系的变换矩阵
    # flange_pose = [275.82, 14.21, 220.41, 150.11, -78.08, 3.99]
    # M_base_flange = robot_pose_to_homogeneous_matrix(flange_pose, order='ZYX')
    # print("法兰盘坐标系到世界坐标系的变换矩阵")
    # print(M_base_flange)
    #
    # M_base_board = M_base_flange @ M_flange_board
    # print("标定板坐标系到机械臂基坐标系的变换矩阵")
    # print(M_base_board)
    #
    # # 标定板坐标系到机械臂基坐标系的变换矩阵
    # robot_pos = M_base_flange @ M_flange_board @ test_board_pose
    # # [0 0 0] => [0.40530398 0.12049056 0.17405847]
    # print(f"标定板 => 机械臂 {test_board_pose.flatten()[:3]} => {robot_pos.flatten()[:3]}")
    #
    # # captures_dir = "./captures"
    # # # 确保目录存在
    # # if not os.path.exists(captures_dir):
    # #     logger.warning(f"目录 {captures_dir} 不存在，无法处理图像")
    # # else:
    # #     # 遍历目录下所有以_rgb.png结尾的文件
    # #     for filename in os.listdir(captures_dir):
    # #         if filename.endswith("_rgb.png"):
    # #             rgb_path = os.path.join(captures_dir, filename)
    # #             detect_and_save_corners(rgb_path, check_direction=False)
