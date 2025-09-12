# -*- coding: utf-8 -*-
# @Time       : 2025/9/6 18:00
# @File       : core_components.py.py
# @Description: 包含了所有硬件和模型模块的核心组件的引用
import logging
import os
import time

import cv2
import numpy as np
import torch

from hardware.camera import RealSenseCamera
from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.dataset_processing.grasp import detect_grasps
from yolov8.inference import YOLOv8_Detector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# ============================================================================
# 硬件模块
# ============================================================================

class CameraHandler:
    """摄像头处理模块的占位符"""

    def __init__(self):
        self.camera = None

    def connect(self):
        time.sleep(0.5)
        self.camera = RealSenseCamera()
        self.camera.connect()
        logger.info("Camera connected...")
        return True

    def get_frame(self):
        if self.camera is None:
            return self.get_default_frame()
        try:
            info = self.camera.get_image_bundle()
            rgb = info['rgb']
            depth = info['aligned_depth']
            return rgb, depth
        except Exception as e:
            logger.info(e)
            raise e

    @staticmethod
    def get_default_frame():
        """获取默认的图像帧"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, 'Camera Not Connected', (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        noise = np.random.randint(0, 10, (480, 640, 3), dtype=np.uint8)
        return frame + noise, np.random.rand(480, 640)

    def disconnect(self):
        logger.info("[Placeholder] Camera disconnected.")


class ArmController:
    """Densor机械臂控制器的占位符"""

    def connect(self, ip):
        # TODO: 替换为真实的 DensorRobot 连接代码
        time.sleep(0.5)
        logger.info(f"[Placeholder] Arm connected to {ip}.")
        return True

    def go_home(self):
        logger.info("[Placeholder] Arm going to home position.")
        time.sleep(2)
        return True

    def move_to(self, pose):
        # TODO: 替换为真实的 robot.send_position(pose)
        logger.info(f"[Placeholder] Arm moving to pose: {pose}")
        time.sleep(1.5)
        return True

    def disconnect(self):
        logger.info("[Placeholder] Arm disconnected.")


class GripperController:
    """蓝牙夹爪控制器的占位符"""

    def connect(self, mac_address):
        # TODO: 替换为真实的 GripperControllerWrapper 连接代码
        time.sleep(0.5)
        logger.info(f"[Placeholder] Gripper connected to {mac_address}.")
        return True

    def open(self):
        logger.info("[Placeholder] Gripper opening.")
        time.sleep(0.5)
        return True

    def close(self):
        logger.info("[Placeholder] Gripper closing.")
        time.sleep(0.5)
        return True

    def disconnect(self):
        logger.info("[Placeholder] Gripper disconnected.")


class RobotPlanner:
    """
    新增: 机器人规划器, 封装了完整的抓取动作序列。
    它协调 ArmController 和 GripperController。
    """

    def __init__(self, arm: ArmController, gripper: GripperController):
        self.arm = arm
        self.gripper = gripper

    def execute_grasp_sequence(self, grasp_pose_world, log_callback):
        """
        执行完整的抓取动作序列, 并通过回调函数记录每一步日志。
        :param grasp_pose_world: (x,y,z,rx,ry,rz) 目标抓取位姿
        :param log_callback: 用于发射日志信号的函数
        """
        # TODO: 从 grasp_pose_world 计算 pre-grasp 和 post-grasp 位置
        pre_grasp_pose = list(grasp_pose_world);
        pre_grasp_pose[2] += 50  # 向上50mm

        # 1. 移动到抓取前位置
        log_callback("[信息] 1/5: 移动到抓取点上方...")
        if not self.arm.move_to(pre_grasp_pose): return False

        # 2. 张开夹爪
        log_callback("[信息] 2/5: 张开夹爪...")
        if not self.gripper.open(): return False

        # 3. 下降到抓取位置
        log_callback("[信息] 3/5: 下降至目标...")
        if not self.arm.move_to(grasp_pose_world): return False

        # 4. 闭合夹爪
        log_callback("[信息] 4/5: 闭合夹爪...")
        if not self.gripper.close(): return False

        # 5. 抬升
        log_callback("[信息] 5/5: 抬升物体...")
        if not self.arm.move_to(pre_grasp_pose): return False

        return True


# ============================================================================
# 算法模块模块
# ============================================================================

class DetectionModel:
    """YOLO-World or YOLOv8 目标检测模型"""

    def __init__(self, model_path=None, model_type='yolov8'):
        """
        初始化检测模型
        :param model_path: 模型文件路径
        :param model_type: 模型类型, 'yolov8' 或 'yolo-world'
        """
        if model_path is None:
            model_path = r"D:\PycharmProjects\robotic-grasping\yolov8\runs\detect\train3\weights\best.pt"
        self.model_type = model_type
        self.model_path = model_path
        logger.info(f"[DetectionModel] [{model_type}] Loading detection model from {model_path}")
        if model_path is None:
            raise Exception("Model path is None")
        if self.model_type == 'yolov8':
            self.detector = YOLOv8_Detector(model_path)
            self.detector.load_model()

    def detect(self, image: np.ndarray, text_prompt: str) -> list:
        """
        执行目标检测
        :param image: 输入图像, np.array, shape=(H, W, 3)
        :param text_prompt: 检测提示词
        :return: 检测结果列表
        """
        logger.info(f"[DetectionModel] Detecting '{text_prompt}'...")
        return self.detector.detect(image)


class GraspModel:
    """GOA-Net 抓取姿态估计模型的占位符"""

    def __init__(self):
        self.model_path = r'D:\PycharmProjects\robotic-grasping\logs\20250906_1408_training_cornell_grconvnet_goa_Baseline\best_iou_epoch_21_iou_0.9209'
        if not os.path.exists(self.model_path):
            raise Exception(f"Grasp Model file not found at {self.model_path}")
        self.cam_data = CameraData(include_depth=True, include_rgb=True)
        logger.info('Loading grasp model... ')
        self.model = torch.load(self.model_path)
        # Get the compute device
        self.device = get_device(force_cpu=False)

    def predict(self, rgb, depth):
        """
        预测抓取姿态
        :param rgb: 输入的RGB图像, np.array, shape=(H, W, 3)
        :param depth: 输入的深度图像, np.array, shape=(H, W, 1)
        :return: 预测的抓取姿态
        """

        x, depth_img, rgb_img = self.cam_data.get_data(rgb=rgb, depth=depth)

        # Predict the grasp pose using the saved model
        with torch.no_grad():
            xc = x.to(self.device)
            pred = self.model.predict(xc)

        q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])
        grasps = detect_grasps(q_img, ang_img, width_img)
        return grasps, q_img, ang_img, width_img


class CoordinateTransformer:
    """坐标转换模块"""

    def __init__(self):
        # 相机内参 3*3
        self.camera_matrix = None
        # 相机到机器人的变换矩阵 4*4
        self.M_base_camera = None

    def load_calibration_file(self) -> bool:
        camera_matrix_file = r'D:\PycharmProjects\robotic-grasping\calibrate\calibrate_result\camera_matrix.txt'
        if not os.path.exists(camera_matrix_file):
            # raise Exception(f"Camera Matrix file not found at {camera_matrix_file}")
            logger.info(f"Camera Matrix file not found at {camera_matrix_file}")
            return False
        self.camera_matrix = np.loadtxt(camera_matrix_file, delimiter=' ')
        logger.info("Camera matrix loaded.")

        M_base_camera_file = r'D:\PycharmProjects\robotic-grasping\calibrate\calibrate_result\M_base_camera.txt'
        if not os.path.exists(M_base_camera_file):
            # raise Exception(f"M_base_camera file not found at {M_base_camera_file}")
            logger.info(f"M_base_camera file not found at {M_base_camera_file}")
            return False
        self.M_base_camera = np.loadtxt(M_base_camera_file, delimiter=' ')
        logger.info("M_base_camera loaded.")
        return True

    def transform_pixel_to_camera(self, u, v, depth_value) -> np.ndarray:
        """
        像素坐标 -> 相机坐标
        :param u: 像素坐标u
        :param v: 像素坐标v
        :param depth_value: 深度值 m
        :return: 相机坐标 [x, y, z]
        """
        if self.camera_matrix is None:
            raise Exception("Camera matrix not loaded.")
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        # 计算相机坐标
        Zc = depth_value
        Xc = (u - cx) * Zc / fx
        Yc = (v - cy) * Zc / fy

        return np.array([Xc, Yc, Zc])

    def transform_pixel_to_base(self, u, v, depth_value) -> np.ndarray:
        """"
        像素坐标 -> 机器人世界坐标
        :param u: 像素坐标u
        :param v: 像素坐标v
        :param depth_value: 深度值 m
        :return: 机器人世界坐标 [x, y, z]
        """
        if self.M_base_camera is None:
            raise Exception("M_base_camera not loaded.")
        # 1. 像素坐标 -> 相机坐标 (需要相机内参)
        camera_coordinate = self.transform_pixel_to_camera(u, v, depth_value)
        # 2. 相机坐标 -> 机器人世界坐标 (需要手眼标定矩阵)
        camera_coord_homog = np.append(camera_coordinate, [1]).reshape(4, 1)  # 转换为齐次坐标
        robot_coord = np.dot(self.M_base_camera, camera_coord_homog)
        robot_base_xyz = robot_coord[:3].flatten()  # 移除齐次坐标
        return robot_base_xyz

    def transform_camera_to_base(self, x_c, y_c, z_c) -> np.ndarray:
        """
        相机坐标 -> 机器人世界坐标
        :param x_c: 相机坐标x
        :param y_c: 相机坐标y
        :param z_c: 相机坐标z
        :return: 机器人世界坐标 [x, y, z]
        """
        if self.M_base_camera is None:
            raise Exception("M_base_camera not loaded.")
        # 转换为齐次坐标
        camera_coord_homog = np.append([x_c, y_c, z_c], [1]).reshape(4, 1)
        # 转换到机器人基坐标系
        robot_coord = np.dot(self.M_base_camera, camera_coord_homog)
        robot_base_xyz = robot_coord[:3].flatten()  # 移除齐次坐标
        return robot_base_xyz


class InstructionParser:
    """中文指令解析器的占位符"""

    def parse(self, text):
        # TODO: 实现我们之前讨论的、更强大的基于关键词的解析器
        logger.info(f"[Placeholder] Parsing instruction: '{text}'")
        if "所有" in text or "全部" in text:
            prompt = "bolt"  # 简化处理，假设是bolt
            return [{'prompt': prompt, 'quantity': 'all'}]
        else:
            # 简化处理
            tasks = []
            sub_commands = text.replace("，", ",").split(",")
            for cmd in sub_commands:
                if "红" in cmd:
                    tasks.append({'prompt': 'red block'})
                elif "蓝" in cmd:
                    tasks.append({'prompt': 'blue ball'})
                else:
                    tasks.append({'prompt': 'object'})  # 默认
            return tasks
