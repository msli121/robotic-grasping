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

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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


# 使用示例
if __name__ == "__main__":
    # 处理captures目录下的文件（可根据实际情况修改）
    try:
        source_dir = "./captures"
        # 处理拍摄的照片文件
        process_checkerboard_and_pose_data(source_dir)
    except Exception as e:
        print(f"处理失败: {str(e)}")
