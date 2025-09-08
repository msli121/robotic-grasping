# -*- coding: utf-8 -*-
# @Time       : 2025/8/16 17:39
# @File       : hand_capture_image.py
# @Description: RealSense 相机采集标定板图片 + 机械臂位置记录 + 角点检测
# @Author     : lms
# @Date       : 2025/8/16 17:39

import logging
import os
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np

from PIL import Image, ImageTk

from hardware.camera import RealSenseCamera
from robot.densor_robot import DensorRobot

# 配置日志系统，记录程序运行状态和错误信息
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class VisionRobotSystem:
    """视觉与机械臂控制系统主类，整合相机采集、图像显示、机械臂位置记录和角点检测功能"""

    def __init__(self, root, device_id=246422072474, robot_host="192.168.1.11", robot_port=5002,
                 checkerboard_size=(8, 8)):
        """
        初始化系统
        :param root: Tkinter主窗口对象
        :param device_id: RealSenseSense相机设备ID
        :param robot_host: 机械臂IP地址
        :param robot_port: 机械臂端口号
        :param checkerboard_size: 标定板角点数量
        """
        self.data_save_dir = os.path.join(BASE_DIR, 'data',
                                          f'hand_capture_{time.strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(self.data_save_dir, exist_ok=True)

        self.root = root
        self.root.title("标定板拍照")
        self.root.geometry("1500x800")  # 调整窗口大小，提供更好的显示效果
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)  # 窗口关闭回调

        # 初始化相机和机械臂对象
        self.camera = RealSenseCamera(device_id=device_id)
        self.robot = None
        self.robot_host = robot_host
        self.robot_port = robot_port

        # 角点检测相关参数
        self.checkerboard_size = checkerboard_size
        self.refine_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.corner_detection_enabled = True  # 角点检测状态

        # 图像数据存储
        self.rgb_image = None  # RGB图像数据
        self.depth_image = None  # 深度图像数据（原始数据）
        self.running = False  # 相机运行状态标志
        self.save_count = 0  # 自动获取下一个保存编号

        # 创建GUI界面组件
        self._create_widgets()

        # 启动相机数据获取线程（守护线程，随主程序退出）
        self.camera_thread = threading.Thread(target=self._update_camera, daemon=True)
        self.camera_thread.start()

    def _create_widgets(self):
        """创建所有GUI界面组件"""
        # 主框架，用于布局管理
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 图像展示区（带边框和标题）
        image_frame = ttk.LabelFrame(main_frame, text="图像展示", padding="10")
        image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # RGB图像标签（左侧）
        self.rgb_label = ttk.Label(image_frame, text="等待相机连接...")
        self.rgb_label.pack(side=tk.LEFT, padx=5, fill=tk.BOTH, expand=True)

        # 深度图像标签（右侧）
        self.depth_label = ttk.Label(image_frame, text="等待相机连接...")
        self.depth_label.pack(side=tk.RIGHT, padx=5, fill=tk.BOTH, expand=True)

        # 功能按钮区
        button_frame = ttk.LabelFrame(main_frame, text="功能控制", padding="10")
        button_frame.pack(fill=tk.X, pady=(0, 10))

        # 连接/断开相机按钮
        self.camera_btn = ttk.Button(button_frame, text="连接相机", command=self._toggle_camera)
        self.camera_btn.pack(side=tk.LEFT, padx=5)

        # 连接/断开机械臂按钮
        self.robot_btn = ttk.Button(button_frame, text="连接机械臂", command=self._toggle_robot)
        self.robot_btn.pack(side=tk.LEFT, padx=5)

        # 角点检测按钮
        self.corner_btn = ttk.Button(button_frame, text="开启角点检测", command=self._toggle_corner_detection)
        self.corner_btn.pack(side=tk.LEFT, padx=5)

        # 拍照按钮（初始禁用，相机连接后启用）
        self.capture_btn = ttk.Button(button_frame, text="拍照", command=self._capture_images, state=tk.DISABLED)
        self.capture_btn.pack(side=tk.LEFT, padx=5)

        # 状态显示栏（底部，显示系统状态）
        self.status_var = tk.StringVar(value="状态: 系统初始化完成")
        self.status_label = ttk.Label(
            main_frame,
            textvariable=self.status_var,
            relief=tk.SUNKEN,  # 凹陷效果
            anchor=tk.W  # 左对齐
        )
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

    def _toggle_camera(self):
        """切换相机连接状态（连接/断开）"""
        if not self.running:
            # 尝试连接相机
            try:
                self.camera.connect()
                self.running = True
                self.camera_btn.config(text="断开相机")
                self.capture_btn.config(state=tk.NORMAL)  # 启用拍照按钮
                self.status_var.set("状态: 相机已连接，正在获取图像...")
                logger.info("相机连接成功")
            except Exception as e:
                messagebox.showerror("相机连接错误", f"无法连接相机: {str(e)}")
                logger.error(f"相机连接错误: {e}")
        else:
            # 断开相机连接
            self.running = False
            self.camera.disconnect()  # 断开相机连接
            self.camera_btn.config(text="连接相机")
            self.capture_btn.config(state=tk.DISABLED)  # 禁用拍照按钮
            self.status_var.set("状态: 相机已断开")
            logger.info("相机已断开")

    def _toggle_robot(self):
        """切换机械臂连接状态（连接/断开）"""
        if self.robot is None or not self._is_robot_connected():
            # 尝试连接机械臂
            try:
                self.robot = DensorRobot(host=self.robot_host, port=self.robot_port)
                self.robot_btn.config(text="断开机械臂")
                self.status_var.set("状态: 机械臂已连接")
                logger.info("机械臂连接成功")
            except Exception as e:
                messagebox.showerror("机械臂连接错误", f"无法连接机械臂: {str(e)}")
                logger.error(f"机械臂连接错误: {e}")
                self.robot = None
        else:
            # 断开机械臂连接
            try:
                self.robot.close()
                self.robot = None
                self.robot_btn.config(text="连接机械臂")
                self.status_var.set("状态: 机械臂已断开")
                logger.info("机械臂已断开")
            except Exception as e:
                logger.error(f"断开机械臂连接时出错: {e}")

    def _toggle_corner_detection(self):
        """切换角点检测状态（开启/关闭）"""
        self.corner_detection_enabled = not self.corner_detection_enabled
        if self.corner_detection_enabled:
            self.corner_btn.config(text="关闭角点检测")
            self.status_var.set("状态: 角点检测已开启")
            logger.info("角点检测已开启")
        else:
            self.corner_btn.config(text="开启角点检测")
            self.status_var.set("状态: 角点检测已关闭")
            logger.info("角点检测已关闭")

    def _is_robot_connected(self):
        """检查机械臂是否处于连接状态"""
        try:
            return self.robot and self.robot.is_connected()
        except:
            return False

    def _detect_checkerboard_corners(self, rgb_image, check_direction=False):
        """
        检测标定板角点
        :param rgb_image: RGB格式的图像
        :return: 原始图像和带有角点的图像（如果检测到角点），否则返回原始图像和None
        """
        # 转换为BGR格式（OpenCV默认格式）
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        # 转换为灰度图
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

        # 寻找角点
        checkerboard_found, corners = cv2.findChessboardCorners(
            gray_image,
            self.checkerboard_size,
            None,
            cv2.CALIB_CB_ADAPTIVE_THRESH
        )

        if checkerboard_found:
            # 优化角点坐标
            corners_refined = cv2.cornerSubPix(
                gray_image,
                corners,
                self.checkerboard_size,
                (-1, -1),
                self.refine_criteria
            )

            # 检查角点方向
            if check_direction and abs(corners[0][0][1] - corners[1][0][1]) > 10:
                logger.warning("角点识别方向与实际方向不一致，跳过处理")
                return rgb_image, None

            # 绘制角点
            bgr_with_corners = bgr_image.copy()
            cv2.drawChessboardCorners(
                bgr_with_corners,
                self.checkerboard_size,
                corners_refined,
                checkerboard_found
            )

            # 转换回RGB格式
            rgb_with_corners = cv2.cvtColor(bgr_with_corners, cv2.COLOR_BGR2RGB)
            return rgb_image, rgb_with_corners
        else:
            logger.info("未检测到标定板角点")
            return rgb_image, None

    def _update_camera(self):
        """相机图像获取与更新线程函数"""
        while True:
            if self.running:
                try:
                    # 获取相机图像数据
                    image_bundle = self.camera.get_image_bundle()
                    self.rgb_image = image_bundle['rgb']  # RGB格式（适合显示）
                    self.depth_image = image_bundle['aligned_depth']  # 深度数据（单位：米）

                    # 处理RGB图像：根据角点检测状态决定是否显示角点
                    if self.corner_detection_enabled:
                        _, display_rgb = self._detect_checkerboard_corners(self.rgb_image)
                        # 如果没有检测到角点，使用原始图像
                        if display_rgb is None:
                            display_rgb = self.rgb_image.copy()
                    else:
                        display_rgb = self.rgb_image.copy()

                    # 深度图预处理：归一化并转换为彩色图（便于可视化）
                    depth_normalized = cv2.normalize(
                        self.depth_image.squeeze(),  # 移除单通道维度
                        None,
                        0, 255,
                        cv2.NORM_MINMAX,
                        dtype=cv2.CV_8U  # 转换为8位无符号整数
                    )
                    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)  # 应用彩色映射

                    # 转换为RGB格式以便正确显示
                    depth_colored_rgb = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

                    # 转换为Tkinter支持的图像格式
                    rgb_img = Image.fromarray(display_rgb)
                    depth_img = Image.fromarray(depth_colored_rgb)

                    # 更新标签图像
                    # self.rgb_label.config(image="")  # 清除之前的图像引用
                    # self.depth_label.config(image="")  # 清除之前的图像引用

                    rgb_photo = ImageTk.PhotoImage(image=rgb_img)
                    depth_photo = ImageTk.PhotoImage(image=depth_img)

                    self.rgb_label.config(image=rgb_photo)
                    self.depth_label.config(image=depth_photo)

                    # 保存图像引用，防止被垃圾回收
                    self.rgb_label.image = rgb_photo
                    self.depth_label.image = depth_photo

                    self.status_var.set("状态: 图像获取正常")
                except Exception as e:
                    self.status_var.set(f"状态: 图像获取错误: {str(e)}")
                    logger.error(f"图像获取错误: {e}")
                    time.sleep(1)  # 出错时延迟重试
            time.sleep(0.04)  # 控制图像更新频率（约20fps）

    def _get_next_save_count(self):
        """确定下一个文件保存编号（确保编号连续递增）"""
        count = 1
        # 检查已有文件的最大编号（RGB图）
        while os.path.exists(os.path.join(self.data_save_dir, f"{count:02d}_rgb.png")):
            count += 1
        return count

    def _capture_images(self):
        """保存当前RGB图、深度图、机械臂位置信息和角点图（如果开启）"""
        # 检查相机是否正常工作
        if not self.running or self.rgb_image is None or self.depth_image is None:
            messagebox.showwarning("警告", "没有可用的图像数据，请确保相机已连接并正常工作")
            return

        try:
            # 确定下一个保存编号
            self.save_count = self._get_next_save_count()

            # 保存RGB图像（转换为BGR格式，符合OpenCV保存要求）
            rgb_bgr = cv2.cvtColor(self.rgb_image, cv2.COLOR_RGB2BGR)
            rgb_path = os.path.join(self.data_save_dir, f"{self.save_count:02d}_origin_rgb.png")
            cv2.imwrite(rgb_path, rgb_bgr)

            # 如果开启角点检测，保存带有角点的图像
            corner_saved = False
            if self.corner_detection_enabled:
                _, rgb_with_corners = self._detect_checkerboard_corners(self.rgb_image)
                if rgb_with_corners is not None:
                    corner_bgr = cv2.cvtColor(rgb_with_corners, cv2.COLOR_RGB2BGR)
                    corner_path = os.path.join(self.data_save_dir, f"{self.save_count:02d}_corner_rgb.png")
                    cv2.imwrite(corner_path, corner_bgr)
                    corner_saved = True
                    logger.info(f"已保存角点图像: {corner_path}")

            # 保存原始深度数据（.npy格式，保留真实深度值）
            depth_npy_path = os.path.join(self.data_save_dir, f"{self.save_count:02d}_depth_raw.npy")
            np.save(depth_npy_path, self.depth_image)

            # 保存可视化深度图（.png格式，便于直观查看）
            depth_normalized = cv2.normalize(
                self.depth_image.squeeze(), None, 0, 255,
                cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            depth_png_path = os.path.join(self.data_save_dir, f"{self.save_count:02d}_depth_visual.png")
            cv2.imwrite(depth_png_path, depth_colored)

            # 获取并保存机械臂位置信息
            pos_saved = False
            pos_path = os.path.join(self.data_save_dir, f"{self.save_count:02d}_robot_pose.txt")
            if self._is_robot_connected():
                position = self.robot.get_current_position()
                logger.info(f"已获取机械臂位置: {position}")
                np.savetxt(pos_path, np.round(position, 6), delimiter=' ', fmt='%.6f')
                pos_saved = True
            else:
                with open(pos_path, 'w', encoding='utf-8') as f:
                    f.write(f"")
                logger.warning("机械臂未连接，位置信息文件已创建但内容为空")

            status_msg = f"状态: 已保存第{self.save_count}组数据"
            info_msg = f"已保存第{self.save_count}组数据\n包含：RGB图、深度图（原始+可视化）"

            if pos_saved:
                status_msg += "（包含TCP位置信息）"
                info_msg += "、TCP位置"

            if corner_saved:
                status_msg += "（包含角点图）"
                info_msg += "、角点图"

            self.status_var.set(status_msg)
            messagebox.showinfo("成功", info_msg)
            logger.info(status_msg)

        except Exception as e:
            messagebox.showerror("保存错误", f"保存数据时出错: {str(e)}")
            logger.error(f"保存数据时出错: {e}")

    def on_close(self):
        """窗口关闭时的资源清理"""
        self.running = False  # 停止相机线程
        if self.robot and self._is_robot_connected():
            self.robot.close()  # 关闭机械臂连接
        if self.camera.pipeline:
            self.camera.pipeline.stop()  # 停止相机管道
        logger.info("系统已关闭")
        self.root.destroy()  # 销毁窗口


def main():
    """主函数：创建并运行系统"""
    root = tk.Tk()
    # 初始化系统（可根据实际设备修改参数）
    app = VisionRobotSystem(
        root,
        device_id=246422072474,  # RealSense相机设备ID
        robot_host="192.168.1.11",  # 机械臂IP地址
        robot_port=5002,  # 机械臂端口号
        checkerboard_size=(8, 8)  # 标定板角点数量
    )
    root.mainloop()


if __name__ == "__main__":
    main()
