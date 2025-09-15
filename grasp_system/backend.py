# backend.py (V3.7 - 非阻塞多线程最终修复版)

import time
import cv2
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QTimer  # 引入 QTimer
from core_components import (CameraHandler, ArmController, GripperController, RobotPlanner,
                             DetectionModel, GraspModel, CoordinateTransformer, InstructionParser)
from logger import UILogger


class SystemBackend(QObject):
    """
    系统的后端逻辑，在一个独立的线程中运行
    采用QTimer驱动的非阻塞模式，确保能及时响应UI信号
    """
    # --- 信号定义  ---
    log_signal = pyqtSignal(str)
    main_image_signal = pyqtSignal(dict)
    roi_image_signal = pyqtSignal(object)
    quality_map_signal = pyqtSignal(object)
    angle_map_signal = pyqtSignal(object)
    width_map_signal = pyqtSignal(object)
    device_connection_signal = pyqtSignal(str, bool)
    grasp_enable_signal = pyqtSignal(bool)
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        # 标志位和任务管理
        self._is_paused_for_execution = False
        self.task_queue = []
        self.loop_task_prompt = None
        self.loop_task_remaining_count = 0
        self.initial_loop_count = 0

        # 初始化所有组件 
        self.camera = CameraHandler()
        self.arm = ArmController()
        self.gripper = GripperController()
        self.planner = RobotPlanner(self.arm, self.gripper)
        self.detector = DetectionModel()
        self.grasp_model = GraspModel()
        self.transformer = CoordinateTransformer()
        self.parser = InstructionParser()

        # 存储当前结果 
        self.final_grasp_pose_world = None
        self.current_target_selection_strategy = "置信度最高"

        # 状态标志位
        self.is_detection_enabled = False
        self.is_grasp_enabled = False
        self.current_text_prompt = "螺丝"  # 默认检测目标

        # --- 新增: 定时器将在工作线程中被创建和使用 ---
        self.timer = None

    # ============================================================================
    # 核心改动: 重构主运行逻辑
    # ============================================================================
    @pyqtSlot()
    def run(self):
        """
        这个方法现在只负责初始化工作线程的环境
        它不再包含阻塞的 while 循环，而是创建一个QTimer来驱动周期性任务
        """
        # 在工作线程中创建和启动 QTimer
        self.timer = QTimer()
        self.timer.timeout.connect(self.main_tick)
        self.timer.start(100)  # 每 100 毫秒 (10 FPS) 触发一次 main_tick
        self.log_signal.emit(UILogger.info("后端线程已启动"))

    def main_tick(self):
        """由QTimer周期性调用的“心跳”函数。"""
        # --- 任务处理逻辑 ---
        if not self._is_paused_for_execution:
            # 用于处理自动抓取任务队列
            if self.loop_task_remaining_count > 0:
                self.process_single_grasp_cycle(is_loop_task=True)
            elif self.task_queue:
                self.process_single_grasp_cycle(is_loop_task=False)

        # 持续更新视频流与可视化
        rgb_frame, depth_frame = self.camera.get_frame()
        if rgb_frame is None:
            # 发送一个空帧的信号，避免UI卡住
            self.main_image_signal.emit({'frame': None, 'detections': [], 'grasps': []})
            return

        display_data = {'frame': rgb_frame, 'detections': [], 'grasps': []}

        # 根据UI开关状态执行相应逻辑
        if self.is_detection_enabled:
            detections = self.detector.detect(rgb_frame, self.current_text_prompt, threshold=0.2)
            display_data['detections'] = detections

            if self.is_grasp_enabled and detections:
                target_detection = self.select_target(detections, depth_frame)
                display_data['grasps'] = self._get_grasp_predictions_from_roi(rgb_frame, depth_frame,
                                                                              target_detection.get('bbox'))

        elif self.is_grasp_enabled:
            # 仅开启抓取预测，则对全图进行操作
            h, w = rgb_frame.shape[:2]
            full_image_bbox = (0, 0, w, h)
            display_data['grasps'] = self._get_grasp_predictions_from_roi(rgb_frame, depth_frame, full_image_bbox)

        self.main_image_signal.emit(display_data)

    def _get_grasp_predictions_from_roi(self, full_rgb, full_depth, detection_bbox):
        """从一个检测框ROI中，提取抓取区域，进行预测，并返回在原图坐标系下的抓取结果。"""
        img_h, img_w = full_rgb.shape[:2]

        x1, y1, x2, y2 = detection_bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        side_length = int(max(w, h, 224))

        grasp_region_x1 = int(cx - side_length / 2)
        grasp_region_y1 = int(cy - side_length / 2)

        # 边界检查
        grasp_region_x1 = np.clip(grasp_region_x1, 0, img_w - side_length)
        grasp_region_y1 = np.clip(grasp_region_y1, 0, img_h - side_length)
        grasp_region_x2 = grasp_region_x1 + side_length
        grasp_region_y2 = grasp_region_y1 + side_length

        grasp_crop_rgb = full_rgb[grasp_region_y1:grasp_region_y2, grasp_region_x1:grasp_region_x2]
        grasp_crop_depth = full_depth[grasp_region_y1:grasp_region_y2,
                           grasp_region_x1:grasp_region_x2] if full_depth is not None else None

        if grasp_crop_rgb.size == 0:
            return []

        # 抓取预测
        grasps_in_crop, q_img, ang_img, width_img = self.grasp_model.predict(grasp_crop_rgb, grasp_crop_depth)

        self.quality_map_signal.emit(q_img)
        self.angle_map_signal.emit(ang_img)
        self.width_map_signal.emit(width_img)
        self.roi_image_signal.emit(grasp_crop_rgb)

        final_grasps = []
        for g in grasps_in_crop:
            # 直接将抓取矩形的四个点加上偏移即可
            points_in_crop = g.get('points')
            points_in_full_img = points_in_crop.copy()
            points_in_full_img[:, 0] += grasp_region_x1  # Y 偏移
            points_in_full_img[:, 1] += grasp_region_y1  # X 偏移

            # 假设 g 对象有 quality 属性
            center = g.get('center')
            quality = q_img[int(center[0]), int(center[1])]

            final_grasps.append({
                'points': points_in_full_img.tolist(),
                'quality': quality
            })
        return final_grasps

    # ============================================================================
    # 任务处理和槽函数
    # ============================================================================
    def process_single_grasp_cycle(self, is_loop_task):
        if is_loop_task:
            task_prompt = self.loop_task_prompt
            log_prefix = f"任务 {self.initial_loop_count - self.loop_task_remaining_count + 1}/{self.initial_loop_count}"
            self.log_signal.emit(UILogger.system(f"{log_prefix}: 开始抓取 '{task_prompt}'..."))
        else:
            task_prompt = self.task_queue[0]['prompt']
            self.log_signal.emit(UILogger.system(f"开始任务: 抓取 '{task_prompt}'..."))

        rgb, depth = self.camera.get_frame()
        self.log_signal.emit(UILogger.info("正在扫描场景寻找目标..."))
        detections = self.detector.detect(rgb, task_prompt)

        if not detections:
            self.log_signal.emit(UILogger.warning("未在当前视野中找到目标"))
            if is_loop_task:
                self.loop_task_remaining_count -= 1
            else:
                self.task_queue.pop(0)
            time.sleep(1)  # 暂停一秒，避免在找不到目标时快速刷屏
            return

        self.log_signal.emit(UILogger.success(f"找到 {len(detections)} 个匹配目标"))

        target_bbox, target_score = self.select_target(detections, depth)
        self.log_signal.emit(UILogger.info(f"应用策略 '{self.current_target_selection_strategy}'，已锁定目标"))

        # ... (后续的抓取计算和坐标转换逻辑也完全不变)
        x1, y1, x2, y2 = target_bbox
        roi = rgb[y1:y2, x1:x2]
        self.roi_image_signal.emit(roi)
        self.log_signal.emit(UILogger.info("正在计算最佳抓取姿态..."))

        q_img, ang_img, width_img = self.grasp_model.predict(roi)
        self.quality_map_signal.emit(q_img)
        self.angle_map_signal.emit(ang_img)
        self.width_map_signal.emit(width_img)

        grasp_v_roi, grasp_u_roi = np.unravel_index(np.argmax(q_img), q_img.shape)
        self.log_signal.emit(UILogger.success("抓取姿态计算完成"))

        grasp_u_img, grasp_v_img = grasp_u_roi + x1, grasp_v_roi + y1
        depth_value = depth[grasp_v_img, grasp_u_img] if depth is not None else 0
        self.final_grasp_pose_world = self.transformer.transform_pixel_to_base(grasp_u_img, grasp_v_img, depth_value)
        self.log_signal.emit(UILogger.success("3D坐标转换完成"))

        self._is_paused_for_execution = True
        self.grasp_enable_signal.emit(True)
        self.log_signal.emit(UILogger.warning("目标已锁定，请点击 '执行抓取'"))

    def select_target(self, detections: list, depth_map):
        # score 排序
        detections = sorted(detections, key=lambda x: x.get('score'), reverse=True)
        return detections[0]

    # --- 槽函数 (Slots) ---
    @pyqtSlot(bool)
    def set_detection_enabled(self, enabled):
        self.is_detection_enabled = enabled
        self.log_signal.emit(UILogger.info(f"目标识别已 {'开启' if enabled else '关闭'}"))

    @pyqtSlot(bool)
    def set_grasp_enabled(self, enabled):
        self.is_grasp_enabled = enabled
        self.log_signal.emit(UILogger.info(f"抓取预测已 {'开启' if enabled else '关闭'}"))

    @pyqtSlot()
    def connect_camera(self):
        print("DEBUG: connect_camera slot has been successfully triggered!")
        self.log_signal.emit(UILogger.info("正在连接相机..."))
        ok = self.camera.connect()
        self.device_connection_signal.emit("cam", ok)
        self.log_signal.emit(UILogger.success("相机连接成功") if ok else UILogger.error("相机连接失败"))
        if ok:
            cal_ok = self.transformer.load_calibration_file()
            if cal_ok:
                self.log_signal.emit(UILogger.error("手眼标定矩阵加载失败！"))
            else:
                self.log_signal.emit(UILogger.success("手眼标定矩阵加载成功"))

    @pyqtSlot()
    def disconnect_camera(self):
        self.camera.disconnect()
        self.device_connection_signal.emit("cam", False)
        self.log_signal.emit(UILogger.info("相机已断开"))

    @pyqtSlot()
    def connect_arm(self):
        ok = self.arm.connect("192.168.1.11")
        self.device_connection_signal.emit("arm", ok)
        self.log_signal.emit(UILogger.success("机械臂连接成功") if ok else UILogger.error("机械臂连接失败"))

    @pyqtSlot()
    def disconnect_arm(self):
        self.arm.disconnect()
        self.device_connection_signal.emit("arm", False)
        self.log_signal.emit(
            UILogger.info("机械臂已断开"))

    @pyqtSlot()
    def connect_gripper(self):
        ok = self.gripper.connect("EC:XX:XX:XX:XX")
        self.device_connection_signal.emit("gripper", ok)
        self.log_signal.emit(
            UILogger.success("夹爪连接成功") if ok else UILogger.error("夹爪连接失败"))

    @pyqtSlot()
    def disconnect_gripper(self):
        self.gripper.disconnect()
        self.device_connection_signal.emit("gripper", False)
        self.log_signal.emit(
            UILogger.info("夹爪已断开"))

    @pyqtSlot(str)
    def set_selection_strategy(self, strategy):
        self.current_target_selection_strategy = strategy
        self.log_signal.emit(
            UILogger.info(f"目标选择策略已更改为: {strategy}"))

    @pyqtSlot(str)
    def process_instruction(self, text):
        self.stop_all_tasks()
        self.log_signal.emit(UILogger.system(f"收到指令: '{text}'"))
        tasks = self.parser.parse(text)
        if not tasks:
            self.log_signal.emit(UILogger.error("指令解析失败"))
            return
        # 更新当前检测文本，即使在非任务模式下也生效
        self.current_text_prompt = tasks[0].get('prompt', '')
        self.log_signal.emit(UILogger.info(f"当前识别目标已设为: '{self.current_text_prompt}'"))

        # rgb, _ = self.camera.get_frame()
        # detections = self.detector.detect(rgb, self.current_text_prompt)
        # count = len(detections)
        # if count > 0:
        #     self.log_signal.emit(UILogger.success(f"扫描到 {count} 个目标"))
        # else:
        #     self.log_signal.emit(UILogger.warning("未扫描到任何目标"))

        # if tasks[0].get('quantity') == 'all':
        #     rgb, _ = self.camera.get_frame()
        #     detections = self.detector.detect(rgb, tasks[0]['prompt'])
        #     count = len(detections)
        #     if count > 0:
        #         self.loop_task_prompt = tasks[0]['prompt']
        #         self.loop_task_remaining_count = count
        #         self.initial_loop_count = count
        #         self.log_signal.emit(UILogger.success(f"扫描到 {count} 个目标，循环任务已启动"))
        #     else:
        #         self.log_signal.emit(UILogger.warning("未扫描到任何目标"))
        # else:
        #     self.task_queue = tasks
        #     self.log_signal.emit(UILogger.success(f"已创建 {len(tasks)} 个任务"))

    @pyqtSlot()
    def execute_grasp(self):
        self.grasp_enable_signal.emit(False)
        self.log_signal.emit(UILogger.system("开始执行物理抓取..."))
        success = self.planner.execute_grasp_sequence(self.final_grasp_pose_world, lambda msg: self.log_signal.emit(
            UILogger.info(msg.replace("[信息] ", ""))))
        self.log_signal.emit(UILogger.success("物理抓取完成") if success else UILogger.error("物理抓取失败！"))
        if self.loop_task_remaining_count > 0:
            self.loop_task_remaining_count -= 1
            if self.loop_task_remaining_count == 0:
                self.log_signal.emit(UILogger.system("'抓取所有'任务已全部完成！"))
                self.loop_task_prompt = None
        elif self.task_queue:
            self.task_queue.pop(0)
            if not self.task_queue: self.log_signal.emit(UILogger.system("所有任务已完成！"))
        self._is_paused_for_execution = False
        self.final_grasp_pose_world = None

    @pyqtSlot()
    def stop_all_tasks(self):
        self.task_queue.clear()
        self.loop_task_prompt = None
        self.loop_task_remaining_count = 0
        self._is_paused_for_execution = False
        self.grasp_enable_signal.emit(False)
        self.log_signal.emit(UILogger.warning("所有任务已强制停止"))

    @pyqtSlot()
    def cleanup(self):
        """停止定时器并准备退出"""
        self.log_signal.emit(UILogger.info("正在停止后端并清理资源..."))
        if self.timer:
            self.timer.stop()
        self.camera.disconnect()
        self.arm.disconnect()
        self.gripper.disconnect()
        # 发射finished信号，通知主线程可以安全退出
        self.finished.emit()
