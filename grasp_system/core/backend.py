import yaml
import logging
import numpy as np

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QTimer

from grasp_system.core.core_components import (CameraHandler, RobotArmController, GripperController, RobotPlanner,
                                               DetectionModel, GraspModel, CoordinateTransformer,
                                               InstructionParser, PostProcessor)
from grasp_system.ui.ui_logger import UILogger

logger = logging.getLogger(__name__)


class SystemBackend(QObject):
    """
    系统的后端逻辑，在一个独立的线程中运行。
    采用QTimer驱动的非阻塞模式，统一处理实时显示和任务驱动的抓取流程
    """
    # --- 信号定义 ---
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

        # --- 加载配置 ---
        try:
            with open("./config/config.yaml", 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            self.log_signal.emit(UILogger.error(f"系统错误: 无法加载配置文件!"))
            raise e

        # 定时器用于驱动非阻塞的实时显示和任务处理
        self.timer = None
        # --- 任务管理 ---
        self._is_paused_for_execution = False  # 是否正在等待用户点击“执行抓取”
        self.task_queue = []  # 存储待执行的查询计划 (QueryPlan)

        # --- 状态标志位 (由UI直接控制) ---
        self.is_detection_enabled = False
        self.is_grasp_enabled = False
        self.current_mode = self.config['application']['detection'][
            'default_mode']  # 默认模式: "closed_set" or "open_vocab"
        self.current_text_prompt = ""  # 指令模式下的实时文本
        self.current_selection_strategy = "置信度最高"  # 默认策略

        # --- 初始化所有核心组件 ---
        # 硬件模块
        self.camera = CameraHandler(width=self.config['hardware']['camera']['resolution'][0],
                                    height=self.config['hardware']['camera']['resolution'][1])
        self.arm = RobotArmController(ip=self.config['hardware']['arm']['ip'],
                                      port=self.config['hardware']['arm']['port'],
                                      home_pose=self.config['hardware']['arm']['home_pose'],
                                      place_target_pose=self.config['hardware']['arm']['place_target_pose'], )
        self.gripper = GripperController(mac_address=self.config['hardware']['gripper']['mac_address'], )
        self.planner = RobotPlanner(self.arm, self.gripper)
        # 算法模块
        self.grasp_model = GraspModel(model_path=self.config['paths']['grasp_model'],
                                      input_size=self.config['application']['grasping']['model_input_size'])
        self.coord_transformer = CoordinateTransformer(camera_matrix_path=self.config['paths']['camera_matrix'],
                                                       M_base_camera_path=self.config['paths']['M_base_camera'])
        self.instruction_parser = InstructionParser(vocab_path=self.config['paths']['vocab'])
        self.post_processor = PostProcessor()
        self.open_vocab_detector = DetectionModel(model_path=self.config['paths']['open_vocab_model'],
                                                  model_type='yoloe-pf')
        self.closed_set_detector = DetectionModel(model_path=self.config['paths']['closed_set_model'],
                                                  model_type='yoloe')
        self.detector = self.closed_set_detector  # 默认激活闭集模型

        # 存储最终计算出的世界坐标抓取位姿
        self.final_grasp_pose_world = None
        # 存储当前检测到的目标角度
        self.final_grasp_pose_angle = None

    @pyqtSlot()
    def run(self):
        """初始化工作线程环境，并启动定时器。"""
        # 加载模型
        try:
            self.log_signal.emit(UILogger.info("正在加载模型..."))
            self.grasp_model.load()
            self.open_vocab_detector.load()
            self.closed_set_detector.load()
            self.coord_transformer.load()
            self.log_signal.emit(UILogger.info("模型加载完成"))
        except Exception as e:
            self.log_signal.emit(UILogger.error(f"模型加载失败: {e}"))
            return

        self.timer = QTimer()
        self.timer.timeout.connect(self.main_tick)
        self.timer.start(self.config['application']['timer_interval_ms'])  # 10 FPS
        self.log_signal.emit(UILogger.info("后端线程已启动"))

    def main_tick(self):
        """
        由QTimer周期性调用的“心跳”函数
        后端逻辑的主循环驱动源
        """
        try:
            # 1. 检查是否有任务需要执行
            if self.task_queue and not self._is_paused_for_execution:
                # 如果队列中有任务，并且系统不处于暂停状态，则执行一个任务周期
                self.process_single_task_cycle()
                return  # 执行完任务周期后，直接返回，等待下一个tick来刷新UI

            # 2. 如果没有任务，则执行实时显示逻辑
            rgb_frame, depth_frame = self.camera.get_frame()
            if rgb_frame is None:
                self.main_image_signal.emit({'frame': None, 'detections': [], 'grasps': []})
                return

            display_data = {'frame': rgb_frame, 'detections': [], 'grasps': []}

            # 识别处理
            if self.is_detection_enabled:
                detections = self.detector.detect(rgb_frame, text_prompt=self.current_text_prompt)
                if self.task_queue:
                    constraints = self.task_queue[0].get('constraints', [])
                    # 约束后处理
                    target = self.post_processor.select_best_target(detections, constraints, rgb_frame, depth_frame)
                    if target:
                        detections = [target]
                    else:
                        detections = []

                display_data['detections'] = detections
                # 抓取预测
                if self.is_grasp_enabled and detections:
                    target_detection = detections[0]
                    grasps, roi, q, ang, w = self._get_grasp_predictions_from_roi(rgb_frame, depth_frame,
                                                                                  target_detection['bbox'])
                    display_data['grasps'] = grasps
                    # 实时更新分析面板
                    self.roi_image_signal.emit(roi)
                    self.quality_map_signal.emit(q)
                    self.angle_map_signal.emit(ang)
                    self.width_map_signal.emit(w)

            elif self.is_grasp_enabled:
                # 只开启抓取，则对全图进行
                h, w = rgb_frame.shape[:2]
                grasps, roi, q, ang, w = self._get_grasp_predictions_from_roi(rgb_frame, depth_frame, (0, 0, w, h))
                display_data['grasps'] = grasps
                # 实时更新分析面板
                self.roi_image_signal.emit(roi)
                self.quality_map_signal.emit(q)
                self.angle_map_signal.emit(ang)
                self.width_map_signal.emit(w)

            self.main_image_signal.emit(display_data)
        except Exception as e:
            logger.error(e)
            self.log_signal.emit(UILogger.error(f"系统错误: {str(e)}"))

    def _get_grasp_predictions_from_roi(self, full_rgb, full_depth, detection_bbox):
        """
        从一个检测框ROI中，提取抓取区域，进行预测，并返回在原图坐标系下的抓取结果和中间图像。
        :param full_rgb: 输入的RGB图像 (H, W, 3)
        :param full_depth: 输入的深度图像 (H, W, 1)
        :param detection_bbox: 检测框的坐标 (x1, y1, x2, y2)
        :return: 包含抓取结果、ROI、质量图、角度图、宽度图的元组
        """
        """从一个检测框ROI中，提取抓取区域，进行预测，并返回在原图坐标系下的抓取结果和中间图像。"""
        img_h, img_w = full_rgb.shape[:2]

        # 创建一个以bbox为中心的，边长为bbox最大边长的正方形抓取区域
        x1, y1, x2, y2 = detection_bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        w, h = x2 - x1, y2 - y1
        side_length = int(max(w, h, 224))  # 保证最小尺寸为224

        grasp_region_x1 = int(np.clip(cx - side_length // 2, 0, img_w - side_length))
        grasp_region_y1 = int(np.clip(cy - side_length // 2, 0, img_h - side_length))
        grasp_region_x2 = grasp_region_x1 + side_length
        grasp_region_y2 = grasp_region_y1 + side_length

        roi_rgb = full_rgb[grasp_region_y1:grasp_region_y2, grasp_region_x1:grasp_region_x2]
        roi_depth = full_depth[grasp_region_y1:grasp_region_y2, grasp_region_x1:grasp_region_x2]

        if roi_rgb.size == 0:
            return [], roi_rgb, None, None, None

        # 抓取预测
        grasps_in_roi, q_img, ang_img, width_img = self.grasp_model.predict(roi_rgb, roi_depth)

        grasps_in_full = []
        for g in grasps_in_roi:
            # 将抓取矩形的四个点加上偏移，转换回原图坐标系
            points_in_crop = g.get('points')
            if points_in_crop is None:
                continue

            points_in_full_img = points_in_crop.copy()
            points_in_full_img[:, 0] += grasp_region_x1  # X 偏移
            points_in_full_img[:, 1] += grasp_region_y1  # Y 偏移

            # (x, y) in list
            center = g.get('center')
            # quality = q_img[int(center[0]), int(center[1])] if q_img is not None else 0

            final_center_x = center[0] + grasp_region_x1
            final_center_y = center[1] + grasp_region_y1

            grasps_in_full.append({
                'points': points_in_full_img.tolist(),  # 原图坐标系下，点坐标为 (x,y)
                'quality': g.get('quality'),
                'angle': g.get('angle'),
                'width': g.get('width'),
                'length': g.get('length'),
                'center': [final_center_x, final_center_y]  # 原图坐标系下，点坐标为 (x,y)
            })
        return grasps_in_full, roi_rgb, q_img, ang_img, width_img

    def process_single_task_cycle(self):
        """
        执行一个完整的“任务驱动”的抓取周期。
        这个函数只在有明确任务时被调用。
        """
        if not self.task_queue:
            return
        plan = self.task_queue[0]
        self.current_text_prompt = plan.get('prompt_for_model', '')

        rgb, depth = self.camera.get_frame()
        self.log_signal.emit(UILogger.system(f"开始任务: 识别 '{self.current_text_prompt}'"))

        # 1. 初步识别
        if plan['mode'] == 'closed_set':
            all_dets = self.detector.detect(rgb_image=rgb)  # 闭集模式不依赖prompt
            if plan.get('class_id_filter') is not None:
                detections = [d for d in all_dets if d.get('class_id') == plan['class_id_filter']]
            else:
                detections = all_dets
        else:  # open_vocab
            detections = self.detector.detect(rgb_image=rgb, text_prompt=self.current_text_prompt)

        if not detections:
            self.log_signal.emit(UILogger.warning("未找到任何候选目标，任务失败"))
            self.task_queue.pop(0)
            return

        self.log_signal.emit(UILogger.success(f"找到 {len(detections)} 个候选目标"))

        # 2. 决策：后处理与最终目标选择
        self.log_signal.emit(UILogger.info(f"正在根据约束进行决策..."))
        best_target_detection = self.post_processor.select_best_target(detections, plan['constraints'], rgb, depth)

        if not best_target_detection:
            self.log_signal.emit(UILogger.warning("所有候选目标均不满足约束条件，任务失败"))
            self.task_queue.pop(0)
            return

        self.log_signal.emit(UILogger.success("已锁定抓取目标！"))

        # 3. 抓取规划
        target_bbox = best_target_detection['bbox']
        grasps, roi, q_img, ang_img, width_img = self._get_grasp_predictions_from_roi(rgb, depth, target_bbox)

        if not grasps:
            self.log_signal.emit(UILogger.error("在锁定目标上未能计算出抓取姿态，任务失败"))
            self.task_queue.pop(0)
            return

        best_grasp = max(grasps, key=lambda g: g['quality'])

        # 4. 坐标转换
        # 从 best_grasp 中提取中心点(u,v)和深度值
        grasp_v, grasp_u = best_grasp['center']
        depth_val = depth[grasp_v, grasp_u].flatten()
        self.final_grasp_pose_world = self.coord_transformer.transform_pixel_to_base(grasp_u, grasp_v, depth_val)
        # [可选] 坐标误差优化
        self.final_grasp_pose_world = self.coord_transformer.optimize_base_pose(self.final_grasp_pose_world)
        # 存储当前检测到的目标角度
        self.final_grasp_pose_angle = best_grasp['angle']
        self.log_signal.emit(UILogger.success(
            f"机械臂3D坐标转换完成 坐标：{self.final_grasp_pose_world} 角度：{self.final_grasp_pose_angle}"))

        # 5. 等待执行
        self._is_paused_for_execution = True
        self.grasp_enable_signal.emit(True)
        self.log_signal.emit(UILogger.warning("任务已准备就绪，请点击 '执行抓取'"))

    # ============================================================================
    # 槽函数 (Slots) - 负责响应UI事件
    # ============================================================================
    @pyqtSlot(bool)
    def set_detection_enabled(self, enabled):
        self.is_detection_enabled = enabled
        self.log_signal.emit(UILogger.info(f"实时目标识别已 {'开启' if enabled else '关闭'}"))

    @pyqtSlot(bool)
    def set_grasp_enabled(self, enabled):
        self.is_grasp_enabled = enabled
        self.log_signal.emit(UILogger.info(f"实时抓取预测已 {'开启' if enabled else '关闭'}"))

    @pyqtSlot(str)
    def set_mode(self, mode_text):  # e.g., "自动模式" or "指令模式"
        self.current_mode = "open_vocab" if "开放" in mode_text else "closed_set"
        self.log_signal.emit(UILogger.info(f"系统模式已切换为: {mode_text} ({self.current_mode})"))
        # TODO: Here you could switch the loaded model in self.detector
        # self.detector.set_strategy(...)

    @pyqtSlot(str)
    def set_selection_strategy(self, strategy):
        self.current_selection_strategy = strategy
        self.log_signal.emit(UILogger.info(f"目标选择策略已更改为: {strategy}"))

    # --- 响应UI的槽函数 ---
    @pyqtSlot(str)
    def set_mode(self, mode_name):  # "open_vocab" or "closed_set"
        self.current_mode = mode_name
        if mode_name == "open_vocab":
            self.detector = self.open_vocab_detector
            self.log_signal.emit(UILogger.info("识别模式已切换为: 开放词汇 (灵活)"))
        else:
            self.detector = self.closed_set_detector
            self.log_signal.emit(UILogger.info("识别模式已切换为: 闭集专家 (高精度)"))

    @pyqtSlot(str)
    def process_instruction(self, text):
        self.stop_all_tasks()
        plan = self.instruction_parser.parse(text, self.current_mode)
        if plan:
            self.task_queue.append(plan)
            self.log_signal.emit(UILogger.success(f"指令解析成功，已创建任务"))
            # # 立即触发任务执行
            # self.process_single_task_cycle()
        else:
            self.log_signal.emit(UILogger.error(f"指令解析失败！"))

    @pyqtSlot()
    def execute_grasp(self):
        # 前置条件校验
        arm_ready = self.arm.connected
        gripper_ready = self.gripper.connected
        if not arm_ready:
            self.log_signal.emit(UILogger.error("机械臂未连接，无法执行抓取"))
            return
        if not gripper_ready:
            self.log_signal.emit(UILogger.error("夹爪未连接，无法执行抓取"))
            return

        self.grasp_enable_signal.emit(False)
        self.log_signal.emit(UILogger.system("开始执行抓取..."))

        success = self.planner.execute_grasp_sequence(
            self.final_grasp_pose_world,
            log_callback=lambda msg: self.log_signal.emit(UILogger.info(msg.replace("[信息] ", "")))
        )

        self.log_signal.emit(UILogger.success("抓取完成") if success else UILogger.error("抓取失败！"))

        # 任务完成，从队列中移除
        if self.task_queue: self.task_queue.pop(0)

        self._is_paused_for_execution = False
        self.final_grasp_pose_world = None

        # # 自动开始下一个任务 (如果存在)
        # if self.task_queue and not self._is_paused_for_execution:
        #     self.process_single_task_cycle()

    @pyqtSlot()
    def stop_all_tasks(self):
        self.task_queue.clear()
        self._is_paused_for_execution = False
        self.grasp_enable_signal.emit(False)
        self.log_signal.emit(UILogger.warning("所有任务已清空，并停止"))

    @pyqtSlot()
    def clear_instruction_context(self):
        """
        响应UI的“清空指令”信号。
        负责重置与上一条指令相关的后端状态。
        """
        self.task_queue.clear()
        self.current_text_prompt = ""
        self._is_paused_for_execution = False
        self.log_signal.emit(UILogger.info("指令已清空"))

    # --- 设备连接槽函数 ---
    @pyqtSlot()
    def connect_camera(self):
        ok = self.camera.connect()
        self.device_connection_signal.emit("cam", ok)
        self.log_signal.emit(UILogger.success("相机连接成功") if ok else UILogger.error("相机连接失败"))
        if ok:
            cal_ok = self.coord_transformer.load()
            self.log_signal.emit(
                UILogger.success("标定矩阵加载成功") if cal_ok else UILogger.error("标定矩阵加载失败！"))

    @pyqtSlot()
    def disconnect_camera(self):
        self.camera.disconnect()
        self.device_connection_signal.emit("cam", False)
        self.log_signal.emit(UILogger.info("相机已断开"))

    @pyqtSlot()
    def connect_arm(self):
        ok = self.arm.connect()
        self.device_connection_signal.emit("arm", ok)
        self.log_signal.emit(UILogger.success("机械臂连接成功") if ok else UILogger.error("机械臂连接失败"))

    @pyqtSlot()
    def disconnect_arm(self):
        self.arm.disconnect()
        self.device_connection_signal.emit("arm", False)
        self.log_signal.emit(UILogger.info("机械臂已断开"))

    @pyqtSlot()
    def connect_gripper(self):
        ok = self.gripper.connect()
        self.device_connection_signal.emit("gripper", ok)
        self.log_signal.emit(UILogger.success("夹爪连接成功") if ok else UILogger.error("夹爪连接失败"))

    @pyqtSlot()
    def disconnect_gripper(self):
        self.gripper.disconnect()
        self.device_connection_signal.emit("gripper", False)
        self.log_signal.emit(UILogger.info("夹爪已断开"))

    @pyqtSlot()
    def cleanup(self):
        """停止定时器并清理资源。"""
        self.log_signal.emit(UILogger.info("正在停止后端并清理资源..."))
        if self.timer:
            self.timer.stop()
        self.camera.disconnect()
        self.arm.disconnect()
        self.gripper.disconnect()
        self.finished.emit()
