# -*- coding: utf-8 -*-
# @Time       : 2025/9/6 18:00
# @File       : core_components.py
# @Description: 包含了所有硬件和模型模块的核心组件
import logging
import math
import os
import time

import cv2
import numpy as np
import yaml

from grasp_predictor import GraspPredictor
from hardware.camera import RealSenseCamera
from robot.densor_robot import DensorRobot
from robot.gripper_controller import GripperControllerWrapper
from yolo.inference import YOLODetector, YOLOEDetector, YOLOETextPromptDetector

logger = logging.getLogger(__name__)


# ============================================================================
# 硬件模块
# ============================================================================

class CameraHandler:
    """摄像头处理模块的占位符"""

    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.camera = None

    def connect(self) -> bool:
        try:
            self.camera = RealSenseCamera(width=self.width, height=self.height)
            self.camera.connect()
            logger.info("Camera connected...")
            return True
        except Exception as e:
            logger.error(f"[Camera] Failed to connect camera: {e}")
            return False

    def disconnect(self):
        if self.camera:
            self.camera.disconnect()
            logger.info("[Camera] Camera disconnected.")
            self.camera = None

    def get_frame(self) -> tuple[np.ndarray, np.ndarray]:
        """
            获取当前的图像帧
            :return: RGB图像 rgb: np.ndarray [H, W, 3],
                    深度图像 depth: np.ndarray [H, W, 1]
        """
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


class GripperController:
    """蓝牙夹爪控制器"""

    def __init__(self, mac_address="EC:23:06:00:D9:FB"):
        self.mac_address = mac_address
        self.gripper = GripperControllerWrapper(self.mac_address)
        self.connected = False

    def connect(self) -> bool:
        self.connected = self.gripper.connect()
        return self.connected

    def disconnect(self):
        logger.info("[Gripper] Gripper disconnected.")
        self.gripper.disconnect()
        self.connected = False

    def open(self):
        logger.info("[Gripper] Gripper opening...")
        self.gripper.open()
        time.sleep(1.5)
        logger.info("[Gripper] Gripper opened.")

    def close(self):
        logger.info("[Gripper] Gripper closing...")
        self.gripper.close()
        time.sleep(1.5)
        logger.info("[Gripper] Gripper closed.")


class RobotArmController:
    """Densor机械臂控制器"""

    def __init__(self, ip="192.168.1.11", port=5002, home_pose: list = None, place_target_pose: list = None):
        """
        初始化机械臂控制器
        :param ip: 机械臂IP地址
        :param port: 机械臂端口号
        :param home_pose: 机械臂默认位姿 home pose
        :param place_target_pose: 机械臂放置物体的位姿 place pose
        """
        if home_pose is None:
            home_pose = [140, 0, 230.0, -167, 2, 81, 5]
        if place_target_pose is None:
            place_target_pose = [140, 0, 230.0, -167, 2, 81, 5]
        self.ip = ip
        self.port = port
        self.home_pose = home_pose
        self.place_target_pose = place_target_pose
        self.robot = DensorRobot(host=ip, port=port)
        self.connected = False

    def connect(self) -> bool:
        self.connected = self.robot.connect()
        if self.connected:
            logger.info("[RobotArm] Robot arm connected.")
        return self.connected

    def disconnect(self):
        self.robot.close()
        self.connected = False

    def go_home(self, pose=None) -> bool:
        if pose is None:
            pose = self.home_pose
        if pose is None:
            logger.error("[RobotArm] Home pose not set.")
            return False
        logger.info(f"[RobotArm] Arm going to home position: {pose}")
        self.robot.send_position(pose)
        time.sleep(1.5)
        return True

    def move_to(self, pose) -> bool:
        if not pose:
            logger.error("[RobotArm] Move pose not set.")
            return False
        logger.info(f"[RobotArm] Move to {pose}")
        self.robot.send_position(pose)
        time.sleep(1.5)
        return True

    def rotate_relative_angle(self, angle: float = 0.0, j_num: int = 6):
        """
        旋转机械臂的指定关节
        :param angle: 旋转角度，单位度
        :param j_num: 关节编号
        """
        logger.info(f"[RobotArm] Rotate joint {j_num} by {angle} degrees")
        self.robot.rotate_relative_angle(angle, j_num)
        time.sleep(1)
        return True

    def get_current_position(self) -> list:
        """
        获取机器人当前位置
        """
        return self.robot.get_current_position()


class RobotPlanner:
    """
    机器人规划器, 封装了完整的抓取动作序列
    它协调 RobotArmController 和 GripperController。
    """

    def __init__(self, arm: RobotArmController, gripper: GripperController):
        self.arm = arm
        self.gripper = gripper

    def execute_grasp_sequence(self, robot_pose: np.array, angle=0.0, log_callback=None):
        """
        执行完整的抓取动作序列, 并通过回调函数记录每一步日志。
        :param robot_pose: (x,y,z,rx, ry, rz) 抓取点姿态，单位mm
        :param angle: 旋转角度，单位弧度，抓取模型预测出来的角度
        :param log_callback: 用于发射日志信号的函数
        """
        if not len(robot_pose) == 7 or max(robot_pose) < 10:
            log_callback("[错误] 机械臂坐标姿态异常")
            return
        robot_pose = list(robot_pose)
        if not len(robot_pose) == 7 or max(robot_pose) < 10:
            log_callback("[错误] 机械臂坐标姿态异常")
            return
        robot_pose = list(robot_pose)

        robot_pose[2] += 50  # 向上50mm
        # 1. 移动到抓取点上方
        log_callback("[信息] 1/7: 移动到抓取点上方...")
        if not self.arm.move_to(robot_pose):
            log_callback(f"[错误] 1/7: 移动到抓取点上方失败")
            return False

        # 2. 张开夹爪
        log_callback("[信息] 2/7: 张开夹爪...")
        if not self.gripper.open():
            log_callback(f"[错误] 2/7: 张开夹爪失败")
            return False

        # 判断是否需要旋转
        if abs(angle) > 0.1:
            log_callback("[信息] : 旋转角度...")
            # 弧度转度，并且变换方向
            self.arm.rotate_relative_angle(CoordinateTransformer.radian_to_degree(-angle), j_num=6)
            cur_robot_pose = self.arm.get_current_position()
            # 更新机械臂姿态
            robot_pose[4:7] = cur_robot_pose[4:7]

        # 3. 下降到抓取位置
        robot_pose[2] -= 50  # 下降50mm
        log_callback("[信息] 3/7: 下降至目标...")
        if not self.arm.move_to(robot_pose):
            log_callback(f"[错误] 3/7: 下降至目标失败")
            return False

        # 4. 闭合夹爪
        log_callback("[信息] 4/7: 闭合夹爪...")
        if not self.gripper.close():
            log_callback(f"[错误] 4/7: 闭合夹爪失败")
            return False

        # 5. 抬升
        log_callback("[信息] 5/7: 抬升物体...")
        if not self.arm.move_to(self.arm.home_pose):
            log_callback(f"[错误] 5/7: 抬升物体失败")
            return False
        if not self.arm.move_to(self.arm.place_target_pose):
            log_callback(f"[错误] 5/7: 移动到放置位置失败")
            return False

        # 6. 张开夹爪
        log_callback("[信息] 6/7: 张开夹爪...")
        if not self.gripper.open():
            log_callback(f"[错误] 6/7: 张开夹爪失败")
            return False

        # 7. 回到home pose
        log_callback("[信息] 7/7: 回到home pose...")
        if not self.arm.go_home():
            log_callback(f"[错误] 7/7: 回到home pose失败")
            return False

        return True


# ============================================================================
# 算法模块模块
# ============================================================================

class DetectionModel:
    """YOLO-World or YOLOv8 目标检测模型"""

    def __init__(self, model_path=None, model_type='yoloe'):
        """
        初始化检测模型
        :param model_path: 模型文件路径
        :param model_type: 模型类型, 'yolo' 或 'yoloe' 或 'yoloe-pf'
        """
        self.model_type = model_type
        self.model_path = model_path
        self.detector = None

    def load(self):
        if self.model_type == 'yolo':
            if not self.model_path:
                self.model_path = r"D:\PycharmProjects\robotic-grasping\yolo\runs\detect\train_yoloe_20250915_5\weights\best.pt"
            logger.info(f"[DetectionModel] [{self.model_type}] Loading detection model from {self.model_path}")
            self.detector = YOLODetector(self.model_path)
            self.detector.load_model()
        elif self.model_type == 'yoloe':
            if not self.model_path:
                self.model_path = r"D:\PycharmProjects\robotic-grasping\yolo\pretrained_models\yoloe-11s-seg.pt"
            logger.info(f"[DetectionModel] [{self.model_type}] Loading detection model from {self.model_path}")
            self.detector = YOLOEDetector(self.model_path)
            self.detector.load_model()
        elif self.model_type == 'yoloe-pf':
            if not self.model_path:
                self.model_path = r"D:\PycharmProjects\robotic-grasping\yolo\pretrained_models\yoloe-11s-seg-pf.pt"
            logger.info(f"[DetectionModel] [{self.model_type}] Loading detection model from {self.model_path}")
            self.detector = YOLOETextPromptDetector(self.model_path)
            self.detector.load_model()

    def detect(self, rgb_image: np.ndarray, text_prompt: str | list[str] | None = None, threshold=0.5) -> list:
        """
        执行目标检测
        :param rgb_image: 输入图像, np.array, shape=(H, W, 3)
        :param text_prompt: 检测提示词
        :param threshold: 置信度阈值
        :return: 检测结果列表
        """
        # logger.info(f"[DetectionModel] Detecting '{text_prompt}'...")
        class_names = None
        if isinstance(text_prompt, str) and text_prompt:
            class_names = [text_prompt]
        elif isinstance(text_prompt, list) and text_prompt:
            class_names = text_prompt
        return self.detector.detect(image=rgb_image, class_names=class_names, threshold=threshold)


class GraspModel:
    """GOA-Net 抓取姿态估计模型的占位符"""

    def __init__(self, model_path=None, input_size=224):
        self.model_path = model_path
        self.input_size = input_size
        self.grasp_predictor = None

    def load(self):
        self.grasp_predictor = GraspPredictor(self.model_path, output_size=self.input_size)
        self.grasp_predictor.load_model()

    def predict(self, rgb, depth):
        """
        预测抓取姿态
        :param rgb: 输入的RGB图像, np.array, shape=(H, W, 3)
        :param depth: 输入的深度图像, np.array, shape=(H, W, 1)
        :return: 预测的抓取姿态
        """
        grasps, q_img, ang_img, width_img = self.grasp_predictor.predict(rgb, depth)
        return grasps, q_img, ang_img, width_img


class CoordinateTransformer:
    """坐标转换模块"""

    def __init__(self, camera_matrix_path=None, M_base_camera_path=None):
        self.camera_matrix_path = camera_matrix_path
        self.M_base_camera_path = M_base_camera_path
        # 相机内参 3*3
        self.camera_matrix = None
        # 相机到机器人的变换矩阵 4*4
        self.M_base_camera = None

    def load(self) -> bool:
        if not self.camera_matrix_path:
            self.camera_matrix_path = r'D:\PycharmProjects\robotic-grasping\calibrate\calibrate_result\camera_matrix.txt'
        if not os.path.exists(self.camera_matrix_path):
            logger.error(f"Camera Matrix file not found at {self.camera_matrix_path}")
            return False
        else:
            self.camera_matrix = np.loadtxt(self.camera_matrix_path, delimiter=' ')
            logger.info(f"相机内参:\n{self.camera_matrix}")
        if not self.M_base_camera_path:
            self.M_base_camera_path = r'D:\PycharmProjects\robotic-grasping\calibrate\calibrate_result\M_base_camera.txt'
        if not os.path.exists(self.M_base_camera_path):
            logger.error(f"M_base_camera file not found at {self.M_base_camera_path}")
            return False
        else:
            self.M_base_camera = np.loadtxt(self.M_base_camera_path, delimiter=' ')
            logger.info(f"手眼变换矩阵:\n{self.M_base_camera}")
        return True

    def transform_pixel_to_camera(self, u, v, depth_value) -> np.ndarray:
        """
        像素坐标 -> 相机坐标
        :param u: 像素坐标u
        :param v: 像素坐标v
        :param depth_value: 深度值 m
        :return: 相机坐标 [x, y, z] np.array
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
        camera_xyz = np.array([Xc, Yc, Zc])
        return camera_xyz.flatten()

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

    def transform_pixel_to_base(self, u, v, depth_value) -> np.ndarray:
        """"
        像素坐标 -> 机器人世界坐标
        :param u: 像素坐标u
        :param v: 像素坐标v
        :param depth_value: 深度值 m
        :return: 机器人世界坐标 [x, y, z]
        """
        # 1. 像素坐标 -> 相机坐标 (需要相机内参)
        camera_xyz = self.transform_pixel_to_camera(u, v, depth_value)
        # 2. 相机坐标 -> 机器人世界坐标 (需要手眼标定矩阵)
        return self.transform_camera_to_base(camera_xyz[0], camera_xyz[1], camera_xyz[2])

    @staticmethod
    def radian_to_degree(rad):
        """
        将输入的弧度值转换到-pi/2到pi/2范围内，然后转换为对应的度数
        参数:
            rad: 输入的弧度值
        返回:
            float: 转换后的度数，范围在-90到90之间
        异常:
            TypeError: 当输入不是数值类型时抛出
        """
        # 检查输入是否为数值类型
        if not isinstance(rad, (int, float)):
            raise TypeError("输入必须是整数或浮点数")

        # 将弧度转换到-pi/2到pi/2范围
        # 使用公式: φ = θ - 2π × round(θ / π - 0.5)
        mapped_rad = rad - 2 * math.pi * round(rad / math.pi - 0.5)

        # 确保结果在-pi/2到pi/2范围内（处理可能的浮点误差）
        if mapped_rad > math.pi / 2:
            mapped_rad = math.pi - mapped_rad
        elif mapped_rad < -math.pi / 2:
            mapped_rad = -math.pi - mapped_rad
        # 转换为度数
        degree = math.degrees(mapped_rad)
        # 结果四舍五入保留两位小数
        degree = round(degree, 2)
        return degree

    @staticmethod
    def optimize_and_convert2robot_pose(base_xyz: np.ndarray) -> np.ndarray:
        """
        优化并转为机械臂姿态
        :param base_xyz: 原始基座标 [x, y, z] 单位：m
        :return: 优化后的基座标 [x, y, z, rx, ry, rz, config] 单位：mm
        """
        if max(base_xyz) > 1:
            print("Warning: base_xyz values are too large, may be in meters instead of millimeters.")
            raise Exception("base_xyz values are too large, may be in meters instead of millimeters.")
        # 1. 将机械臂基坐标转换为mm
        base_xyz = np.asarray(base_xyz)
        base_xyz_mm = base_xyz * 1000
        # 机械臂活动中心点
        center_point = np.array([220, 0, 0])  # 根据实际情况修改
        # xyz轴的缩放误差
        xyz_scale = np.array([1.1, 0.84, 1.5])  # 根据实际情况修改
        # 应用缩放误差
        optimized_xyz_mm = center_point + (base_xyz_mm - center_point) * xyz_scale
        # z轴限制
        optimized_xyz_mm[2] = max(optimized_xyz_mm[2], -30)
        # y+ 时抓取姿态pose
        y_plus_pose = [-164, 7, 84, 5]
        # y- 时抓取姿态pose
        y_minus_pose = [-163, -5, 83, 5]
        if optimized_xyz_mm[1] < 0:
            optimized_xyz_mm = np.concatenate((optimized_xyz_mm, y_minus_pose))
        else:
            optimized_xyz_mm = np.concatenate((optimized_xyz_mm, y_plus_pose))
        optimized_xyz_mm = optimized_xyz_mm.flatten()
        logger.info(f"优化后的机械臂姿态:{optimized_xyz_mm}")
        return optimized_xyz_mm


class PostProcessor:
    """
    智能决策模块
    - 根据查询计划中的约束，从多个检测结果中筛选出唯一的目标。
    """

    # --- ADDED: 定义颜色HSV范围 ---
    COLOR_HSV_RANGES = {
        'red': [([0, 120, 70], [10, 255, 255]), ([170, 120, 70], [180, 255, 255])],
        'green': [([35, 100, 50], [85, 255, 255])],
        'blue': [([100, 150, 50], [140, 255, 255])],
        'yellow': [([20, 100, 100], [30, 255, 255])],
        'white': [([0, 0, 200], [180, 30, 255])],
        'black': [([0, 0, 0], [180, 255, 50])]
    }

    def __init__(self, coord_transformer: CoordinateTransformer):
        """
        初始化后处理器。
        :param coord_transformer: 一个已初始化的CoordinateTransformer实例，用于3D计算。
        """
        self.coord_transformer = coord_transformer

    def select_best_target(self, detections: list, constraints: list, rgb_image: np.ndarray,
                           depth_image: np.ndarray = None) -> dict | None:
        """
        应用多个约束，通过多轮筛选的方式决策出最佳目标
        :param detections: 检测到的目标列表
        :param constraints: 查询计划中的约束列表
        :param rgb_image: RGB图像
        :param depth_image: 深度图像
        """
        candidates = detections.copy()
        if not candidates:
            return None

        # --- 步骤 1: 应用所有过滤型约束 ---
        filter_constraints = [c for c in constraints if c['type'] == 'color']
        for constraint in filter_constraints:
            color_to_find = constraint['value']
            candidates = [d for d in candidates if self._is_color_dominant(d['bbox'], rgb_image, color_to_find)]
            if not candidates:
                return None

        # --- 步骤 2: 定义排序型约束的优先级和处理函数 ---
        # 优先级从上到下递减。
        SORTING_PRIORITY = [
            'nearest', 'farthest',  # 距离约束最优先
            'largest', 'smallest',  # 其次是尺寸约束
            'topmost', 'bottommost',  # 最后是2D位置约束
            'rightmost', 'leftmost'
        ]

        # 排序函数字典保持不变
        sorters = {
            'largest': lambda d: self._get_bbox_area(d['bbox']),
            'smallest': lambda d: -self._get_bbox_area(d['bbox']),
            'leftmost': lambda d: -(d['bbox'][0] + d['bbox'][2]) / 2,
            'rightmost': lambda d: (d['bbox'][0] + d['bbox'][2]) / 2,
            'topmost': lambda d: -(d['bbox'][1] + d['bbox'][3]) / 2,
            'bottommost': lambda d: (d['bbox'][1] + d['bbox'][3]) / 2,
            'nearest': lambda d: -self._get_3d_distance(d['bbox'], depth_image) if depth_image is not None else -np.inf,
            'farthest': lambda d: self._get_3d_distance(d['bbox'], depth_image) if depth_image is not None else -np.inf
        }

        # 提取指令中出现的所有排序约束
        active_constraints = [c['value'] for c in constraints if c['value'] in sorters]

        # --- 步骤 3: 按照优先级顺序，进行多轮筛选 ---
        for constraint_name in SORTING_PRIORITY:
            if constraint_name in active_constraints:
                # 如果当前优先级的约束在指令中出现了
                if not candidates:
                    break  # 没有候选者直接停止

                # 获取对应的排序函数
                sorter = sorters[constraint_name]

                # 找到当前候选者中的最大值
                # 使用一个小的容差(tolerance)来处理浮点数精度问题，认为相近的值是“并列第一”
                max_val = max(sorter(c) for c in candidates)
                tolerance = 1e-5

                # 选出所有接近最大值的候选者，作为下一轮的输入
                candidates = [c for c in candidates if sorter(c) >= max_val - tolerance]

                # 如果筛选后只剩一个，提前结束
                if len(candidates) == 1:
                    return candidates[0]

        # --- 步骤 4: 处理特殊约束和最终决策 ---
        if 'middle' in [c['value'] for c in constraints]:
            # middle 是一个特例，它需要先按x坐标排序
            candidates.sort(key=lambda d: (d['bbox'][0] + d['bbox'][2]) / 2)
            if candidates:
                return candidates[len(candidates) // 2]
            else:
                return None

        # 如果经过所有排序约束后仍有多个候选者（例如，两个物体完全一样大且并排）
        # 则使用默认的置信度规则来选出最终的一个
        if candidates:
            candidates.sort(key=lambda d: d.get('score', 0), reverse=True)
            return candidates[0]
        else:
            # 如果在某一轮筛选后，候选者列表变空了
            return None

    # --- 辅助函数 ---
    @staticmethod
    def _get_bbox_area(bbox):
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)

    def _is_color_dominant(self, bbox, image, color_name, threshold=0.15):
        """
        检查bbox内是否以指定颜色为主导。
        """
        color_name = color_name.lower()
        if color_name not in self.COLOR_HSV_RANGES:
            return False  # 不支持的颜色

        x1, y1, x2, y2 = map(int, bbox)
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return False

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

        total_mask = np.zeros(hsv_roi.shape[:2], dtype="uint8")

        # 应用该颜色的所有HSV范围
        for (lower, upper) in self.COLOR_HSV_RANGES[color_name]:
            mask_part = cv2.inRange(hsv_roi, np.array(lower), np.array(upper))
            total_mask = cv2.bitwise_or(total_mask, mask_part)

        # 计算颜色像素占总面积的比例
        color_pixel_count = cv2.countNonZero(total_mask)
        total_pixel_count = roi.shape[0] * roi.shape[1]

        return (color_pixel_count / total_pixel_count) > threshold

    def _get_3d_distance(self, bbox, depth_map):
        """
        计算bbox中心的真实3D距离。
        """
        if self.coord_transformer is None or self.coord_transformer.camera_matrix is None:
            # 如果没有坐标转换器，则无法计算3D距离，返回一个极大值
            return np.inf

        x1, y1, x2, y2 = map(int, bbox)
        u, v = (x1 + x2) // 2, (y1 + y2) // 2

        # 从5x5邻域中获取鲁棒的深度值
        patch = depth_map[max(0, v - 2):v + 3, max(0, u - 2):u + 3]
        valid_depths = patch[patch > 0]  # 忽略无效深度

        if valid_depths.size == 0:
            return np.inf  # 如果区域内没有有效深度，则无法计算

        depth_median = np.median(valid_depths)

        try:
            # 转换为相机坐标系
            Xc, Yc, Zc = self.coord_transformer.transform_pixel_to_camera(u, v, depth_median)
            # 计算欧几里得距离
            return np.sqrt(Xc ** 2 + Yc ** 2 + Zc ** 2)
        except Exception:
            return np.inf


class InstructionParser:
    """
    中文指令解析器
    - 在闭集模式下，严格要求指令中包含已知的核心实体。
    - 在开放词汇模式下，更具灵活性。
    """

    def __init__(self, vocab_path="./config/vocab.yaml"):
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                self.vocab = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"指令配置文件未找到: {vocab_path}！请检查路径。")
        # 为了提高查找效率，将词典反转并合并
        self._build_lookup_tables()

    def _build_lookup_tables(self):
        """构建高效的查找表，支持中英文到标准英文的映射。"""
        self.entity_map = {}
        for zh, en in self.vocab['class_map'].items():
            self.entity_map[zh] = (zh, en)
            self.entity_map[en.lower()] = (zh, en)  # 支持英文输入

        self.attribute_map = {}
        for attr_type, attr_dict in self.vocab['attributes'].items():
            for zh, en in attr_dict.items():
                self.attribute_map[zh] = (en, attr_type)
                self.attribute_map[en.lower()] = (en, attr_type)

        self.constraint_map = {}
        for const_type, const_dict in self.vocab['constraints'].items():
            for zh, en in const_dict.items():
                self.constraint_map[zh] = (en, const_type)
                self.constraint_map[en.lower()] = (en, const_type)
        # self.class_map_zh_to_en = {k: v for k, v in self.vocab['class_map'].items()}
        # self.class_map_en_to_zh = {v: k for k, v in self.vocab['class_map'].items()}

        self.class_map_en_to_id = {v: i for i, v in enumerate(self.vocab['class_map'].values())}
        self.class_map_zh_to_id = {k: i for i, k in enumerate(self.vocab['class_map'].keys())}

    def parse(self, text: str, mode: str) -> dict | None:
        """
        解析用户输入的中文文本，生成一个结构化的查询计划
        开集模型只能支持英文指令，
        :param text: 用户输入的中文文本
        :param mode: 解析模式，'closed_set' 或 'open_vocab'
        :return: 结构化的查询计划字典
        """
        plan = {
            "mode": mode,
            "input_text": text,
            "target_entity_en": None,
            "target_entity_zh": None,
            "prompt_for_model": "",
            "class_id_filter": None,  # 仅闭集模式用
            "constraints": []
        }

        # 1. 预处理
        processed_text = text.lower().replace("的", "").replace("抓", "").replace("取", "").strip()

        # 2. 查找核心实体
        found_entity_key = None
        for key in self.entity_map.keys():
            if key in processed_text:
                found_entity_key = key
                plan["target_entity_zh"], plan["target_entity_en"] = self.entity_map[key]
                processed_text = processed_text.replace(key, "").strip()  # 移除实体词
                break

        # 3. 查找属性和约束
        attributes_en = []
        # 按词长排序，优先匹配长词 (例如 "最左边" "最左")
        all_modifiers = sorted(list(self.attribute_map.keys()) + list(self.constraint_map.keys()), key=len,
                               reverse=True)

        for key in all_modifiers:
            if key in processed_text:
                if key in self.attribute_map:
                    en_val, type_val = self.attribute_map[key]
                    attributes_en.append(en_val)
                    plan["constraints"].append({"type": type_val, "value": en_val})
                elif key in self.constraint_map:
                    en_val, type_val = self.constraint_map[key]
                    plan["constraints"].append({"type": type_val, "value": en_val})
                processed_text = processed_text.replace(key, "").strip()

        # 4. 根据模式构建最终plan
        if mode == 'closed_set':
            if not plan["target_entity_en"]:
                logger.error(f"指令解析失败: 在闭集模式下，指令 '{text}' 中未找到已知物体")
                return None
            # 在闭集模式下，我们忽略属性，只关心核心实体
            plan["prompt_for_model"] = plan["target_entity_zh"]
            # 从英文名获取class_id
            plan['class_id_filter'] = self.class_map_zh_to_id.get(plan["target_entity_zh"])
        elif mode == 'open_vocab':
            if plan["target_entity_en"]:
                # 场景1: 找到了已知实体 (例如 "红色的螺丝")
                prompt_parts = attributes_en + [plan["target_entity_en"]]
                plan["prompt_for_model"] = " ".join(prompt_parts)
            elif processed_text:
                # 场景2: 没有找到已知实体，将剩余部分作为未知实体 (例如 "那个订书机")
                # 这里的 `processed_text` 已经是移除了所有已知修饰词后的部分
                unknown_entity = processed_text
                plan["target_entity_zh"] = unknown_entity  # 中文设为原始剩余文本
                plan["target_entity_en"] = unknown_entity  # 英文也暂时设为它 (假设输入是英文或中英混合)

                prompt_parts = attributes_en + [unknown_entity]
                plan["prompt_for_model"] = " ".join(prompt_parts)
            else:
                logger.error(f"指令解析失败: 开放模式下指令 '{text}' 无法识别出任何有效目标")
                return None
        return plan
