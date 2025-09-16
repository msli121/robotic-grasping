# -*- coding: utf-8 -*-
# @Time       : 2025/9/6 18:00
# @File       : core_components.py
# @Description: 包含了所有硬件和模型模块的核心组件
import logging
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

    def connect(self) -> bool:
        return self.gripper.connect()

    def disconnect(self):
        logger.info("[Gripper] Gripper disconnected.")
        self.gripper.disconnect()

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

    def connect(self) -> bool:
        return self.robot.connect()

    def disconnect(self):
        self.robot.close()

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


class RobotPlanner:
    """
    机器人规划器, 封装了完整的抓取动作序列
    它协调 RobotArmController 和 GripperController。
    """

    def __init__(self, arm: RobotArmController, gripper: GripperController):
        self.arm = arm
        self.gripper = gripper

    def execute_grasp_sequence(self, robot_xyz: list, log_callback):
        """
        执行完整的抓取动作序列, 并通过回调函数记录每一步日志。
        :param robot_xyz: list (x,y,z) 目标抓取的xyz坐标，单位m
        :param log_callback: 用于发射日志信号的函数
        """
        robot_xyz = list(robot_xyz)
        # m -> mm
        for i in range(3):
            robot_xyz[i] *= 1000
        robot_pose = robot_xyz + self.arm.home_pose[3:]

        robot_pose[2] += 50  # 向上50mm
        # 1. 移动到抓取前位置
        log_callback("[信息] 1/7: 移动到抓取点上方...")
        if not self.arm.move_to(robot_pose): return False

        # 2. 张开夹爪
        log_callback("[信息] 2/7: 张开夹爪...")
        if not self.gripper.open(): return False

        # 3. 下降到抓取位置
        robot_pose[2] -= 50  # 下降50mm
        log_callback("[信息] 3/7: 下降至目标...")
        if not self.arm.move_to(robot_pose): return False

        # 4. 闭合夹爪
        log_callback("[信息] 4/7: 闭合夹爪...")
        if not self.gripper.close(): return False

        # 5. 抬升
        log_callback("[信息] 5/7: 抬升物体...")
        if not self.arm.move_to(self.arm.place_target_pose): return False

        # 6. 张开夹爪
        log_callback("[信息] 6/7: 张开夹爪...")
        if not self.gripper.open(): return False

        # 7. 回到home pose
        log_callback("[信息] 7/7: 回到home pose...")
        if not self.arm.go_home(): return False

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

    def detect(self, image: np.ndarray, text_prompt: str | list[str] | None = None, threshold=0.5) -> list:
        """
        执行目标检测
        :param image: 输入图像, np.array, shape=(H, W, 3)
        :param text_prompt: 检测提示词
        :param threshold: 置信度阈值
        :return: 检测结果列表
        """
        # logger.info(f"[DetectionModel] Detecting '{text_prompt}'...")
        target_classes = None
        if isinstance(text_prompt, str):
            target_classes = [text_prompt]
        elif isinstance(text_prompt, list):
            target_classes = text_prompt
        return self.detector.detect(image=image, target_classes=target_classes, threshold=threshold)


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
            logger.info(f"Camera matrix loaded: {self.camera_matrix}")
        if not self.M_base_camera_path:
            self.M_base_camera_path = r'D:\PycharmProjects\robotic-grasping\calibrate\calibrate_result\M_base_camera.txt'
        if not os.path.exists(self.M_base_camera_path):
            logger.error(f"M_base_camera file not found at {self.M_base_camera_path}")
            return False
        else:
            self.M_base_camera = np.loadtxt(self.M_base_camera_path, delimiter=' ')
            logger.info(f"M_base_camera loaded: {self.M_base_camera}")
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

    @staticmethod
    def optimize_base_pose(base_xyz: np.ndarray) -> np.ndarray:
        """
        优化基座标，修复误差
        :param base_xyz: 原始基座标 [x, y, z] 单位：m
        :return: 优化后的基座标 [x, y, z] 单位：m
        """
        # 机械臂活动返回中心点
        center_point = np.array([0.22, 0, 0])  # 根据实际情况修改
        # xyz轴的缩放误差
        xyz_scale = np.array([1.11496, 0.8479, 0.9253])  # 根据实际情况修改
        # 应用缩放误差
        optimized_xyz = center_point + (base_xyz - center_point) * xyz_scale
        return optimized_xyz


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
            raise FileNotFoundError(f"词典文件未找到: {vocab_path}！请检查路径。")
        self.class_map_zh_to_en = {k: v for k, v in self.vocab['class_map'].items()}
        self.class_map_en_to_zh = {v: k for k, v in self.vocab['class_map'].items()}

        self.class_map_en_to_id = {v: i for i, v in enumerate(self.vocab['class_map'].values())}
        self.class_map_zh_to_id = {k: i for i, k in enumerate(self.vocab['class_map'].keys())}

    def parse(self, text: str, mode: str) -> dict:
        """
        解析用户输入的中文文本，生成一个结构化的查询计划
        :param text: 用户输入的中文文本
        :param mode: 解析模式，'closed_set' 或 'open_vocab'
        :return: 结构化的查询计划字典
        """
        plan = {"mode": mode, "prompt": "", "class_id_filter": None, "constraints": []}
        original_text = text

        target_entity_zh, target_entity_en = None, None
        attributes_en = []

        # 1. 提取核心实体
        for zh, en in self.vocab['class_map'].items():
            if zh in text:
                target_entity_zh, target_entity_en = zh, en
                text = text.replace(zh, "")  # 移除已处理的核心词
                break

        # 2. 核心实体存在性检查 (根据模式)
        if not target_entity_en:
            if mode == 'closed_set':
                print(f"解析失败: 在闭集模式下，指令 '{original_text}' 中未找到已知的核心物体。")
                return None
            else:  # open_vocab 模式
                # 将清理后的整个指令作为prompt
                plan['prompt'] = original_text.replace("的", "").replace("抓取", "").strip()
                # 开放模式下也需要解析约束
                self._extract_constraints(original_text, plan)
                return plan

        # 3. 提取属性
        for attr_type, attr_dict in self.vocab['attributes'].items():
            for zh, en in attr_dict.items():
                if zh in text:
                    attributes_en.append(en)
                    # 属性也作为一种约束加入，用于后处理
                    plan['constraints'].append({"type": attr_type, "value": en})

        # 4. 构建 Prompt 和 Class ID Filter
        if mode == 'closed_set':
            plan['prompt'] = target_entity_en  # Prompt 只是核心实体
            plan['class_id_filter'] = self.class_map_en_to_id.get(target_entity_en)
        else:  # open_vocab
            # Prompt 是属性和实体的组合
            prompt_parts = attributes_en + [target_entity_en]
            plan['prompt'] = " ".join(prompt_parts)

        # 5. 提取约束
        self._extract_constraints(original_text, plan)

        return plan

    def _extract_constraints(self, text, plan):
        """辅助函数，用于从文本中提取约束。"""
        for const_type, const_dict in self.vocab['constraints'].items():
            for zh, en in const_dict.items():
                if zh in text:
                    plan['constraints'].append({"type": const_type, "value": en})


class PostProcessor:
    """
    智能决策模块
    - 根据查询计划中的约束，从多个检测结果中筛选出唯一的目标。
    """

    def select_best_target(self, detections: list, constraints: list, rgb_image: np.ndarray,
                           depth_image: np = None) -> dict | None:
        """
        应用约束，筛选最佳目标
        :param detections: 检测结果列表，每个元素是一个字典，包含 'bbox', 'score', 'class_id' 等。
        :param constraints: 查询计划中的约束列表，每个元素是一个字典，包含 'type' 和 'value'。
        :param rgb_image: 输入的 RGB 图像，用于颜色筛选。
        :param depth_image: 可选的深度图，用于位置筛选。
        :return: 最佳目标的检测字典，或 None。
        """
        candidates = detections.copy()

        # --- 颜色筛选 ---
        # 颜色约束
        color_constraint = next((c for c in constraints if c['type'] == 'color'), None)
        if color_constraint:
            color_to_find = color_constraint['value']
            candidates = [d for d in candidates if self._is_color_dominant(d['bbox'], rgb_image, color_to_find)]

        if not candidates: return None

        # --- 应用排序型约束 (尺寸、位置) 排序找到最优 ---
        # 尺寸约束
        size_constraint = next((c for c in constraints if c['type'] == 'size'), None)
        if size_constraint:
            is_largest = size_constraint['value'] == 'largest'
            candidates.sort(key=lambda d: self._get_bbox_area(d['bbox']), reverse=is_largest)
            return candidates[0]  # 尺寸约束具有最高优先级，直接返回结果

        # 位置约束
        pos_constraint = next((c for c in constraints if c['type'] == 'position'), None)
        if pos_constraint:
            if pos_constraint['value'] == 'leftmost':
                # 按 bbox 中心 x 坐标排序
                candidates.sort(key=lambda d: (d['bbox'][0] + d['bbox'][2]) / 2)
                return candidates[0]
            elif pos_constraint['value'] == 'rightmost':
                # 按 bbox 中心 x 坐标排序
                candidates.sort(key=lambda d: (d['bbox'][0] + d['bbox'][2]) / 2, reverse=True)
                return candidates[0]
            elif pos_constraint['value'] == 'topmost':
                # 按 bbox 中心 y 坐标排序
                candidates.sort(key=lambda d: (d['bbox'][1] + d['bbox'][3]) / 2)
                return candidates[0]
            elif pos_constraint['value'] == 'bottommost':
                # 按 bbox 中心 y 坐标排序
                candidates.sort(key=lambda d: (d['bbox'][1] + d['bbox'][3]) / 2, reverse=True)
                return candidates[0]
            elif pos_constraint['value'] == 'middle':
                # 按 bbox 中心 x 坐标排序
                candidates.sort(key=lambda d: (d['bbox'][0] + d['bbox'][2]) / 2)
                return candidates[len(candidates) // 2]
            elif pos_constraint['value'] == 'nearest':
                # 按3D距离排序
                candidates.sort(key=lambda d: self._get_3d_distance(d['bbox'], depth_image))
                return candidates[0]

        # 如果没有排序型约束，则默认返回置信度最高的
        candidates.sort(key=lambda d: d['score'], reverse=True)
        return candidates[0]

    # --- 辅助函数 ---
    def _get_bbox_area(self, bbox):
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)

    def _is_color_dominant(self, bbox, image, color_name):
        """一个简化的颜色检查函数 (占位符)。"""
        # TODO: 实现更鲁棒的颜色检测逻辑
        # 例如: 裁剪ROI -> 转换到HSV空间 -> 计算颜色直方图 -> 判断主色调
        print(f"Checking if dominant color is '{color_name}' in bbox {bbox} (Not Implemented)")
        return True  # 暂时总是返回 True

    def _get_3d_distance(self, bbox, depth_map):
        """计算 bbox 中心的3D距离 (占位符)。"""
        # TODO: 需要相机内参才能实现
        # 1. 计算 bbox 中心点 (u, v)
        # 2. 从 depth_map 获取深度 Z
        # 3. (u, v, Z) -> (Xc, Yc, Zc) (相机坐标)
        # 4. 返回 sqrt(Xc^2 + Yc^2 + Zc^2)
        print(f"Calculating 3D distance for bbox {bbox} (Not Implemented)")
        # 暂时用2D面积作为替代来模拟排序
        return -self._get_bbox_area(bbox)
