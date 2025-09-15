# -*- coding: utf-8 -*-
# @Time       : 2025/9/6 18:00
# @File       : core_components.py.py
# @Description: 包含了所有硬件和模型模块的核心组件的引用
import logging
import os
import time

import cv2
import numpy as np
import yaml

from grasp_predictor import GraspPredictor
from hardware.camera import RealSenseCamera
from yolo.inference import YOLODetector

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

    def __init__(self, model_path=None, model_type='yolo'):
        """
        初始化检测模型
        :param model_path: 模型文件路径
        :param model_type: 模型类型, 'yolo' 或 'yolo-world'
        """
        if model_path is None:
            model_path = r"D:\PycharmProjects\robotic-grasping\yolo\runs\detect\train_yoloe_20250915_5\weights\best.pt"
        self.model_type = model_type
        self.model_path = model_path
        self.detector = None
        logger.info(f"[DetectionModel] [{model_type}] Loading detection model from {model_path}")
        if model_path and os.path.exists(model_path):
            if self.model_type == 'yolo':
                self.detector = YOLODetector(model_path)
                self.detector.load_model()

    def detect(self, image: np.ndarray, text_prompt: str, threshold=0.5) -> list:
        """
        执行目标检测
        :param image: 输入图像, np.array, shape=(H, W, 3)
        :param text_prompt: 检测提示词
        :param threshold: 置信度阈值
        :return: 检测结果列表
        """
        # logger.info(f"[DetectionModel] Detecting '{text_prompt}'...")
        target_classes = []
        if text_prompt:
            target_classes = [text_prompt]
        return self.detector.detect(image=image, target_classes=target_classes, threshold=threshold)


class GraspModel:
    """GOA-Net 抓取姿态估计模型的占位符"""

    def __init__(self, model_path=None):
        if model_path is None:
            self.model_path = r'D:\PycharmProjects\robotic-grasping\trained-models\cornell-randsplit-rgbd-grconvnet3-drop1-ch32\epoch_19_iou_0.98'
        self.grasp_predictor = GraspPredictor(self.model_path, output_size=224)
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
    """
    中文指令解析器
    - 在闭集模式下，严格要求指令中包含已知的核心实体。
    - 在开放词汇模式下，更具灵活性。
    """

    def __init__(self, vocab_path="./vocab.yaml"):
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

    def select_best_target(self, detections: list, constraints: list, rgb_image: np, depth_image: np = None):
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