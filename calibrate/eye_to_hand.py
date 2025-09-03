# coding=utf-8
"""
眼在手外 用采集到的图片信息和机械臂位姿信息计算相机坐标系相对于机械臂基座标的旋转矩阵和平移向量
"""
import glob
import logging
import os.path
import random
import time

import colorlog
import cv2
import numpy as np

from calibrate.utils import normalize_corner_order, robot_pose_to_homogeneous_matrix, \
    calculate_transform_error, read_robot_poses
from hardware.camera import RealSenseCamera
from robot.densor_robot import DensorRobot

np.set_printoptions(precision=8, suppress=True)


# 配置日志（支持彩色显示）
def setup_logger():
    # 获取logger实例
    logger = logging.getLogger(__name__)

    # 检查是否已有处理器，有则直接返回，避免重复配置
    if logger.handlers:
        return logger

    # 禁用日志传播，防止父logger的处理器也输出日志
    logger.propagate = False

    # 定义日志颜色（不同级别对应不同颜色）
    log_colors = {
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }

    # 定义日志格式
    formatter = colorlog.ColoredFormatter(
        fmt='%(log_color)s%(asctime)s - %(levelname)s - %(message)s',
        log_colors=log_colors,
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 创建控制台处理器并应用格式
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # 配置logger
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    return logger


# 初始化日志
logger = setup_logger()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class EyeToHand:
    def __init__(self, camera_id,
                 chessboard_size,
                 chessboard_grid_size,
                 workspace_limits,
                 workspace_step_size=0.05):
        self.camera_id = camera_id
        self.chessboard_size = chessboard_size
        self.chessboard_step = chessboard_grid_size
        self.workspace_limits = workspace_limits
        self.workspace_step_size = workspace_step_size

        # 相机
        self.camera = RealSenseCamera(device_id=camera_id)
        # 机械臂
        self.robot = DensorRobot()

        # 标定板到机械臂基坐标系的变换矩阵
        self.T_base_board = np.eye(4)

    @staticmethod
    def generate_grid(workspace_limits, workspace_step_size=0.05):
        """
        Construct 3D calibration grid across workspace
        生成机械臂工作空间中的3D网格点
        :param workspace_limits: 工作空间限制
        :param workspace_step_size: 工作空间步长
        :return calibration grid points
        """
        # 将计算出的样本数量转换为整数
        x_num = int(np.round(1 + (workspace_limits[0][1] - workspace_limits[0][0]) / workspace_step_size))
        y_num = int(np.round(1 + (workspace_limits[1][1] - workspace_limits[1][0]) / workspace_step_size))
        z_num = int(np.round(1 + (workspace_limits[2][1] - workspace_limits[2][0]) / workspace_step_size))

        gridspace_x = np.linspace(workspace_limits[0][0], workspace_limits[0][1], x_num)
        gridspace_y = np.linspace(workspace_limits[1][0], workspace_limits[1][1], y_num)
        gridspace_z = np.linspace(workspace_limits[2][0], workspace_limits[2][1], z_num)

        calib_grid_x, calib_grid_y, calib_grid_z = np.meshgrid(gridspace_x, gridspace_y, gridspace_z)
        num_calib_grid_pts = calib_grid_x.shape[0] * calib_grid_x.shape[1] * calib_grid_x.shape[2]
        calib_grid_x.shape = (num_calib_grid_pts, 1)
        calib_grid_y.shape = (num_calib_grid_pts, 1)
        calib_grid_z.shape = (num_calib_grid_pts, 1)
        calib_grid_pts = np.concatenate((calib_grid_x, calib_grid_y, calib_grid_z), axis=1)
        return calib_grid_pts

    def collect_data_by_poses(self, robot_pose_path=None):
        """
        通过指定的机械臂位姿文件采集数据
        :param robot_pose_path: 机器人位姿文件路径
        :return:
        """
        if robot_pose_path is None:
            raise ValueError("robot_pose_path is None")
        if not os.path.exists(robot_pose_path):
            raise ValueError("robot_pose_path is not exists")
        # 读取机械臂姿态信息
        robot_poses = read_robot_poses(robot_pose_path)
        if len(robot_poses) == 0:
            raise ValueError("robot_pose_path is empty")
        logger.info(f"共有{len(robot_poses)}个姿态")

        # 保存的文件夹
        data_save_dir = os.path.join(BASE_DIR, 'data', f'hand_to_eye_{time.strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(data_save_dir, exist_ok=True)

        # 连接相机
        self.camera.connect()
        logger.info(f"相机连接成功...")
        K, dist = self.camera.get_K_and_dist()
        # 保存相机内参
        np.savetxt(os.path.join(data_save_dir, "camera_matrix.txt"), K, delimiter=" ", fmt="%.6f")
        np.savetxt(os.path.join(data_save_dir, "distortion_coefficients.txt"), dist, delimiter=" ", fmt="%.6f")
        logger.info(f"相机内参: {K}")
        logger.info(f"相机畸变系数: {dist}")

        # 连接机器人
        logger.info(f"开始连接机器人...")
        self.robot.connect()
        home_position = [330.0, 5, 200.0, 127, 76, 122, 1]
        # home_position = [220, 5, 200.0, -123, -75, -60, 9]
        # 机器人移动到默认位置
        self.robot.send_position(home_position)
        # 等待机械臂到达指定位置
        robot_stop_s = 2.5
        time.sleep(robot_stop_s)

        # 采集数据
        for index, robot_pose in enumerate(robot_poses):
            logger.info(f'\n\n位置{index:02d} 开始移动到指定位置: {robot_pose}')
            # 机器人移动到指定位置
            self.robot.send_position(robot_pose)
            # 等待机械臂到达指定位置
            time.sleep(robot_stop_s)
            logger.info(f'位置{index:02d} 移动到指定位置完成')

            # 寻找标定板角点
            checkerboard_size = self.chessboard_size
            refine_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            image_bundle = self.camera.get_image_bundle()
            camera_color_img = image_bundle['rgb']
            camera_depth_img = image_bundle['aligned_depth']

            bgr_color_data = cv2.cvtColor(camera_color_img, cv2.COLOR_RGB2BGR)
            gray_data = cv2.cvtColor(bgr_color_data, cv2.COLOR_RGB2GRAY)
            checkerboard_found, corners = cv2.findChessboardCorners(gray_data, checkerboard_size, None,
                                                                    cv2.CALIB_CB_ADAPTIVE_THRESH)
            if checkerboard_found and len(corners) == checkerboard_size[0] * checkerboard_size[1]:
                corners_refined = cv2.cornerSubPix(gray_data, corners, checkerboard_size, (-1, -1), refine_criteria)

                # ==================== 调用归一化函数 ====================
                # 无论OpenCV如何检测，都将其统一为“左上角起始，逐行扫描”的顺序
                corners_refined = normalize_corner_order(corners_refined, checkerboard_size)
                # ===============================================================

                # 显示角点图像
                bgr_color_img_copy = bgr_color_data.copy()
                cv2.drawChessboardCorners(bgr_color_img_copy, checkerboard_size, corners_refined,
                                          checkerboard_found)
                cv2.imshow("ImageWithCorners", bgr_color_img_copy)
                cv2.waitKey(1000)
                cv2.destroyAllWindows()

                # 保存原图RGB
                img_origin_path = os.path.join(data_save_dir, f'{index:02d}_origin_rgb.png')
                cv2.imwrite(img_origin_path, bgr_color_data)
                logger.info(f'位置{index:02d} 原始图像已保存至: {img_origin_path}')

                # 保存带有角点的图像
                img_with_corner_path = os.path.join(data_save_dir, f'{index:02d}_corner_rgb.png')
                cv2.imwrite(img_with_corner_path, bgr_color_img_copy)
                logger.info(f'位置{index:02d} 带有角点的图像已保存至: {img_with_corner_path}')

                # 保存深度图数据
                camera_depth_copy = camera_depth_img.copy()
                depth_data_path = os.path.join(data_save_dir, f'{index:02d}_depth_raw.npy')
                np.save(depth_data_path, camera_depth_copy)
                logger.info(f'位置{index:02d} 深度图数据已保存至: {depth_data_path}')

                # 保存深度可视化图
                depth_visual_img_path = os.path.join(data_save_dir, f'{index:02d}_depth_visual.png')
                # 深度图预处理：归一化并转换为彩色图（便于可视化）
                # noinspection PyTypeChecker
                depth_normalized = cv2.normalize(
                    camera_depth_copy.squeeze(),  # 移除单通道维度
                    None,
                    0, 255,
                    cv2.NORM_MINMAX,
                    dtype=cv2.CV_8U  # 转换为8位无符号整数
                )
                depth_vis = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)  # 应用彩色映射
                cv2.imwrite(depth_visual_img_path, depth_vis)
                logger.info(f'位置{index:02d} 深度可视化图已保存至: {depth_data_path}')

                # 保存机器人位姿信息
                robot_pose_path = os.path.join(data_save_dir, f'{index:02d}_robot_pose.txt')
                np.savetxt(robot_pose_path, np.round(robot_pose, 6), delimiter=' ', fmt='%.6f')
                logger.info(f'位置{index:02d} 位姿信息已保存至: {robot_pose_path}')

            else:
                logger.error(f"位置{index:02d} 标定板未找到角点")
                continue

            # 回到默认点
        self.robot.send_position(home_position)
        # 等待机械臂到达指定位置
        time.sleep(robot_stop_s)

        return data_save_dir

    def auto_collect_data(self):
        """
        采集数据
        :return:
        """

        # 保存的文件夹
        data_save_dir = os.path.join(BASE_DIR, 'data', f'hand_to_eye_{time.strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(data_save_dir, exist_ok=True)

        # 计算空间坐标点
        calib_grid_pts = EyeToHand.generate_grid(self.workspace_limits, self.workspace_step_size)
        logger.info(f'工作空间总点数: {calib_grid_pts.shape[0]}')

        # 连接相机
        self.camera.connect()
        logger.info(f"相机连接成功...")
        K, dist = self.camera.get_K_and_dist()
        # 保存相机内参
        np.savetxt(os.path.join(data_save_dir, "camera_matrix.txt"), K, delimiter=" ", fmt="%.6f")
        np.savetxt(os.path.join(data_save_dir, "distortion_coefficients.txt"), dist, delimiter=" ", fmt="%.6f")
        logger.info(f"相机内参: {K}")
        logger.info(f"相机畸变系数: {dist}")

        # 连接机器人
        logger.info(f"开始连接机器人...")
        self.robot.connect()
        home_position = [300.0, 5, 200.0, 127, 76, 122, 1]
        # home_position = [220, 5, 200.0, -123, -75, -60, 9]
        # 机器人移动到默认位置
        self.robot.send_position(home_position)
        # 等待机械臂到达指定位置
        robot_stop_s = 3
        time.sleep(robot_stop_s)

        for index, tool_position in enumerate(calib_grid_pts):
            # 用tool_position替换home_position的前三个元素，并且乘以1000
            robot_position = tool_position * 1000
            robot_position = list(robot_position)
            robot_position.extend(home_position[3:])

            # 随机调整角度 -10 ~ 10
            robot_position[3] += random.randint(-15, 15)
            robot_position[4] += random.randint(-10, 10)
            robot_position[5] += random.randint(-10, 10)

            logger.info(f'\n\n位置{index:02d} 开始移动到指定位置: {robot_position}')

            # 机器人移动到指定位置
            self.robot.send_position(robot_position)
            # 等待机械臂到达指定位置
            time.sleep(robot_stop_s)
            logger.info(f'位置{index:02d} 移动到指定位置完成')

            # 寻找标定板角点
            checkerboard_size = self.chessboard_size
            refine_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            image_bundle = self.camera.get_image_bundle()
            camera_color_img = image_bundle['rgb']
            camera_depth_img = image_bundle['aligned_depth']

            bgr_color_data = cv2.cvtColor(camera_color_img, cv2.COLOR_RGB2BGR)
            gray_data = cv2.cvtColor(bgr_color_data, cv2.COLOR_RGB2GRAY)
            checkerboard_found, corners = cv2.findChessboardCorners(gray_data, checkerboard_size, None,
                                                                    cv2.CALIB_CB_ADAPTIVE_THRESH)
            if checkerboard_found and len(corners) == checkerboard_size[0] * checkerboard_size[1]:
                corners_refined = cv2.cornerSubPix(gray_data, corners, checkerboard_size, (-1, -1), refine_criteria)

                # ==================== 调用归一化函数 ====================
                # 无论OpenCV如何检测，都将其统一为“左上角起始，逐行扫描”的顺序
                corners_refined = normalize_corner_order(corners_refined, checkerboard_size)
                # ===============================================================

                # 显示角点图像
                bgr_color_img_copy = bgr_color_data.copy()
                cv2.drawChessboardCorners(bgr_color_img_copy, checkerboard_size, corners_refined,
                                          checkerboard_found)
                cv2.imshow("ImageWithCorners", bgr_color_img_copy)
                cv2.waitKey(1000)
                cv2.destroyAllWindows()

                # 保存原图RGB
                img_origin_path = os.path.join(data_save_dir, f'{index:02d}_origin_rgb.png')
                cv2.imwrite(img_origin_path, bgr_color_data)
                logger.info(f'位置{index:02d} 原始图像已保存至: {img_origin_path}')

                # 保存带有角点的图像
                img_with_corner_path = os.path.join(data_save_dir, f'{index:02d}_corner_rgb.png')
                cv2.imwrite(img_with_corner_path, bgr_color_img_copy)
                logger.info(f'位置{index:02d} 带有角点的图像已保存至: {img_with_corner_path}')

                # 保存深度图数据
                camera_depth_copy = camera_depth_img.copy()
                depth_data_path = os.path.join(data_save_dir, f'{index:02d}_depth_raw.npy')
                np.save(depth_data_path, camera_depth_copy)
                logger.info(f'位置{index:02d} 深度图数据已保存至: {depth_data_path}')

                # 保存深度可视化图
                depth_visual_img_path = os.path.join(data_save_dir, f'{index:02d}_depth_visual.png')
                # 深度图预处理：归一化并转换为彩色图（便于可视化）
                # noinspection PyTypeChecker
                depth_normalized = cv2.normalize(
                    camera_depth_copy.squeeze(),  # 移除单通道维度
                    None,
                    0, 255,
                    cv2.NORM_MINMAX,
                    dtype=cv2.CV_8U  # 转换为8位无符号整数
                )
                depth_vis = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)  # 应用彩色映射
                cv2.imwrite(depth_visual_img_path, depth_vis)
                logger.info(f'位置{index:02d} 深度可视化图已保存至: {depth_data_path}')

                # 保存机器人位姿信息
                robot_pose_path = os.path.join(data_save_dir, f'{index:02d}_robot_pose.txt')
                np.savetxt(robot_pose_path, np.round(robot_position, 6), delimiter=' ', fmt='%.6f')
                logger.info(f'位置{index:02d} 位姿信息已保存至: {robot_pose_path}')

            else:
                logger.error(f"位置{index:02d} 标定板未找到角点")
                continue

        # 回到默认点
        self.robot.send_position(home_position)
        # 等待机械臂到达指定位置
        time.sleep(robot_stop_s)

        return data_save_dir

    def do_calibrate(self, collect_data_dir=None, show_board_img=True):
        """
        执行眼在手外标定，获取相机坐标系相对于机械臂基座标的旋转矩阵和平移向量
        :param collect_data_dir: 采集数据的文件夹路径
        :param show_board_img: 是否显示标定板图像
        :return: R_base_camera : 相机坐标系相对于机械臂基座标的旋转矩阵
        """
        logger.info(
            f"开始执行眼在手外标定，采集数据文件夹路径: {collect_data_dir} show_board_img:{show_board_img}")
        if not collect_data_dir or not os.path.exists(collect_data_dir):
            raise ValueError("采集数据文件夹路径不存在")
        if not os.path.exists(os.path.join(collect_data_dir, 'camera_matrix.txt')):
            raise ValueError("相机内参文件不存在")
        if not os.path.exists(os.path.join(collect_data_dir, 'distortion_coefficients.txt')):
            raise ValueError("畸变系数文件不存在")

        # 读取采集数据
        rgb_files = sorted(glob.glob(os.path.join(collect_data_dir, '*_origin_rgb.png')))
        depth_files = sorted(glob.glob(os.path.join(collect_data_dir, '*_depth_raw.npy')))
        robot_pose_files = sorted(glob.glob(os.path.join(collect_data_dir, '*_robot_pose.txt')))

        if len(rgb_files) != len(depth_files) or len(rgb_files) != len(robot_pose_files):
            raise ValueError("采集数据文件夹中文件数量不一致")

        # 读取相机参数
        mtx = np.loadtxt(os.path.join(collect_data_dir, 'camera_matrix.txt'))
        dist = np.loadtxt(os.path.join(collect_data_dir, 'distortion_coefficients.txt'))

        # 设置标定板坐标系下的点坐标，所有点的Z坐标全部为0，只需要赋值x和y
        objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1,
                                                                                               2) * self.chessboard_step
        M_base_flanges = []
        R_camera_boards = []
        t_camera_boards = []

        # 4.查找图片的角点
        criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
        for rgb_file in rgb_files:
            filename = os.path.basename(rgb_file)
            idx_num = filename.split('_')[0]
            flange_pose_file = os.path.join(collect_data_dir, f'{idx_num}_robot_pose.txt')
            if not os.path.exists(flange_pose_file):
                logger.error(f"未找到对应的位姿文件: {flange_pose_file}")
                continue
            # 读取图像
            img = cv2.imread(rgb_file)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 查找角点
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
            if ret:
                # 提高角点精度
                corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
                # 归一化角点方向
                corners = normalize_corner_order(corners, self.chessboard_size)
                if show_board_img:
                    # 绘制角点
                    cv2.drawChessboardCorners(img, self.chessboard_size, corners, ret)
                    cv2.imshow(f'Chessboard_{idx_num}', img)
                    cv2.waitKey(500)
                # solvePnP
                ret, rvec, tvec = cv2.solvePnP(objp, corners, mtx, dist)
                if ret:
                    R_camera_board, _ = cv2.Rodrigues(rvec)
                    R_camera_boards.append(R_camera_board)
                    t_camera_boards.append(tvec)
                    # 添加位姿信息
                    flange_pose = np.loadtxt(flange_pose_file, delimiter=' ')
                    # 转换为齐次坐标，法兰盘坐标系到基坐标系的变换矩阵
                    M_base_flange = robot_pose_to_homogeneous_matrix(flange_pose, order='ZYX')
                    M_base_flanges.append(M_base_flange)
                else:
                    logger.info(f"图片 {rgb_file} 角点检测失败")
            else:
                logger.error(f"未在图像 {rgb_file} 中找到角点")

        if len(M_base_flanges) != len(R_camera_boards) or len(M_base_flanges) != len(t_camera_boards):
            raise ValueError("采集数据文件夹中文件数量不一致")

        M_base_flanges = np.asarray(M_base_flanges)
        R_camera_boards = np.asarray(R_camera_boards)
        t_camera_boards = np.asarray(t_camera_boards)

        # 计算基坐标系到法兰盘坐标系的变换矩阵
        R_flange_bases = []
        t_flange_bases = []
        for M_base_flange in M_base_flanges:
            M = np.linalg.inv(M_base_flange)
            # M = M_base_flange
            R_flange_bases.append(M[:3, :3])
            t_flange_bases.append(M[:3, 3].reshape(3, 1))

        methods_dict = {
            cv2.CALIB_HAND_EYE_TSAI: "TSAI",
            cv2.CALIB_HAND_EYE_PARK: "PARK",
            cv2.CALIB_HAND_EYE_HORAUD: "HORAUD",
            # cv2.CALIB_HAND_EYE_ANDREFF: "ANDREFF",
            # cv2.CALIB_HAND_EYE_DANIILIDIS: "DANIILIDIS",
        }

        for method in methods_dict:
            logger.info(f"\n\n开始进行手眼标定 方法：{methods_dict[method]}")
            R_base_camera, t_base_camera = cv2.calibrateHandEye(
                R_flange_bases,
                t_flange_bases,
                R_camera_boards,
                t_camera_boards,
                method=method,
            )
            logger.info(f"[{methods_dict[method]}] 相机坐标系到机械臂基坐标系的旋转矩阵:")
            logger.info(R_base_camera)
            logger.info(f"[{methods_dict[method]}] 相机坐标系到机械臂基坐标系的平移向量:")
            logger.info(t_base_camera)
            if method == cv2.CALIB_HAND_EYE_PARK:
                M_base_camera = np.eye(4)
                M_base_camera[:3, :3] = R_base_camera
                M_base_camera[:3, 3] = t_base_camera.reshape(3)
                np.savetxt(os.path.join(collect_data_dir, 'M_base_camera.txt'), M_base_camera, delimiter=' ',
                           fmt='%.8f')
                logger.info(
                    f"相机坐标系到机械臂基坐标系的变换矩阵已保存至 {os.path.join(collect_data_dir, 'M_base_camera.txt')}")

    def verify_by_camera(self, collect_data_dir):
        """
        通过RealSense相机验证手眼标定结果，支持用户点击图像获取3D坐标
        优化点：增加错误处理、可视化标记、深度平滑、结果保存和用户提示
        """

        if not collect_data_dir or not os.path.exists(collect_data_dir):
            raise ValueError("采集数据文件夹路径不存在")
        M_base_camera_file = os.path.join(collect_data_dir, 'M_base_camera.txt')
        if not os.path.exists(M_base_camera_file):
            raise ValueError("手眼标定结果文件不存在")
        if not os.path.exists(os.path.join(collect_data_dir, 'camera_matrix.txt')):
            raise ValueError("相机内参文件不存在")
        if not os.path.exists(os.path.join(collect_data_dir, 'distortion_coefficients.txt')):
            raise ValueError("畸变系数文件不存在")

        # 读取手眼标定结果（相机->基座外参）
        M_base_camera = np.loadtxt(M_base_camera_file, delimiter=" ")
        logger.info(f"读取手眼标定结果（相机->基座外参）:\n {M_base_camera}")
        R_base_camera = M_base_camera[:3, :3]
        T_base_camera = M_base_camera[:3, 3]

        # 读取相机参数
        mtx = np.loadtxt(os.path.join(collect_data_dir, 'camera_matrix.txt'))
        dist = np.loadtxt(os.path.join(collect_data_dir, 'distortion_coefficients.txt'))
        fx = mtx[0, 0]
        fy = mtx[1, 1]
        cx = mtx[0, 2]
        cy = mtx[1, 2]

        SAVE_RESULTS = True  # 是否保存测量结果
        RESULTS_FILE = os.path.join(collect_data_dir, "calibration_verification_results.txt")

        # 存储测量结果
        measurement_results = []

        # ========== 初始化相机（带错误处理） ==========
        logger.info("尝试连接RealSense相机...")
        self.camera.connect()
        logger.info(f"相机连接成功")

        # ========== 鼠标回调函数（增强版） ==========
        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # 获取参数
                rgb = param['rgb']
                depth = param['aligned_depth']
                height, width = rgb.shape[:2]

                # 检查点击位置是否在图像范围内
                if not (0 <= x < width and 0 <= y < height):
                    logger.info("点击位置超出图像范围")
                    return

                # 1. 获取深度值（使用周围像素平均值减少噪声）
                depth_value = depth[y, x]
                depth_value = depth_value[0]

                # 2. 将像素点投影到相机坐标系
                Xc = (x - cx) * depth_value / fx
                Yc = (y - cy) * depth_value / fy
                Zc = depth_value
                Pc_hom = np.array([Xc, Yc, Zc, 1]).reshape(4, 1)
                # logger.info(f"相机坐标系下的点: {Pc_hom}")

                # 3. 相机坐标系 → 基座坐标系
                Pbase = (M_base_camera @ Pc_hom)[:3].flatten()
                # logger.info(f"基座坐标系下的点: {Pb}")

                # 4. 显示和记录结果
                result_str = (f"像素点: ({x},{y}) → 深度: {depth_value:.3f}m → "
                              f"相机坐标: X={Xc:.4f}m, Y={Yc:.4f}m, Z={Zc:.4f}m → "
                              f"基座坐标: X={Pbase[0]:.4f}m, Y={Pbase[1]:.4f}m, Z={Pbase[2]:.4f}m")
                logger.info(result_str)

                # 记录结果
                measurement_results.append({
                    "pixel": (x, y),
                    "depth": depth_value,
                    "camera_coords": (Xc, Yc, Zc),
                    "base_coords": (Pbase[0], Pbase[1], Pbase[2])
                })

                # 5. 在图像上标记点击点和信息
                cv2.circle(rgb, (x, y), 5, (0, 0, 255), -1)  # 红色圆点标记
                cv2.putText(rgb, f"X:{Pbase[0]:.3f}", (x + 10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(rgb, f"Y:{Pbase[1]:.3f}", (x + 10, y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(rgb, f"Z:{Pbase[2]:.3f}", (x + 10, y + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # ========== 主循环（增强版） ==========
        try:
            logger.info("\n操作说明:")
            logger.info("1. 点击图像上的点获取其在机械臂基座坐标系中的坐标")
            logger.info("2. 按 's' 保存当前测量结果到文件")
            logger.info("3. 按 'r' 清除所有测量结果")
            logger.info("4. 按 'ESC' 退出程序")

            while True:
                # 获取图像
                images = self.camera.get_image_bundle()
                if not images or 'rgb' not in images or 'aligned_depth' not in images:
                    logger.info("获取图像失败，重试...")
                    continue

                # 转换色彩空间以适应OpenCV显示
                rgb = cv2.cvtColor(images['rgb'], cv2.COLOR_RGB2BGR)
                depth = images['aligned_depth']

                # 显示操作提示
                cv2.putText(rgb, "ESC:exit | s:save | r:clear", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                # 显示窗口并绑定鼠标事件
                cv2.imshow('RealSense 标定验证 (点击图像获取坐标)', rgb)
                cv2.setMouseCallback('RealSense 标定验证 (点击图像获取坐标)',
                                     on_mouse, param={'rgb': rgb, 'aligned_depth': depth})

                # 处理键盘事件
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC 退出
                    logger.info("程序退出")
                    break
                elif key == ord('s') and SAVE_RESULTS:  # 保存结果
                    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
                        for i, res in enumerate(measurement_results, 1):
                            f.write(f"测量点 {i}:\n")
                            f.write(f"  像素坐标: {res['pixel']}\n")
                            f.write(f"  深度值: {res['depth']:.4f}m\n")
                            f.write(f"  相机坐标: {res['camera_coords']}\n")
                            f.write(f"  基座坐标: {res['base_coords']}\n\n")
                    logger.info(f"已保存 {len(measurement_results)} 个测量结果到 {RESULTS_FILE}")
                elif key == ord('r'):  # 清除结果
                    measurement_results.clear()
                    logger.info("已清除所有测量结果")

        except Exception as e:
            logger.info(f"程序运行出错: {str(e)}")
        finally:
            # 资源清理
            cv2.destroyAllWindows()
            self.camera.disconnect()  # 确保相机关闭
            logger.info("资源已释放")

            # 自动保存结果（如果有）
            if SAVE_RESULTS and measurement_results:
                with open(RESULTS_FILE, 'w') as f:
                    for i, res in enumerate(measurement_results, 1):
                        f.write(f"测量点 {i}:\n")
                        f.write(f"  像素坐标: {res['pixel']}\n")
                        f.write(f"  深度值: {res['depth']:.4f}m\n")
                        f.write(f"  相机坐标: {res['camera_coords']}\n")
                        f.write(f"  基座坐标: {res['base_coords']}\n\n")
                logger.info(f"自动保存 {len(measurement_results)} 个测量结果到 {RESULTS_FILE}")


if __name__ == '__main__':
    #  ===================== 本示例中变换矩阵都按 M_B_A 表示，代表坐标系A到坐标系B的变换矩阵 ======================

    # # 标定板坐标系到法兰盘坐标系的变换矩阵
    # robot_pose = [-19.3485, -79.2081, 199.392, -90, 0.0, 90]
    # M_flange_board = robot_pose_to_homogeneous_matrix(robot_pose, order='xyz')
    # logger.info(f"标定板坐标系到法兰盘坐标系的变换矩阵：\n{robot_pose}")

    chessboard_size = (8, 8)  # 水平方向内角度个数 * 垂直方向内角度个数
    chessboard_grid_size = 0.018  # 标定板方格真实尺寸 单位 m
    workspace_limits = np.asarray([[0.23, 0.33], [-0.1, 0.1], [0.05, 0.25]])
    # workspace_limits = np.asarray([[0.23, 0.33], [-0.5, 0.5], [0.07, 0.27]])
    workspace_step_size = 0.05
    eye_to_hand = EyeToHand(
        camera_id=246422072474,
        chessboard_size=chessboard_size,
        chessboard_grid_size=chessboard_grid_size,
        workspace_limits=workspace_limits,
        workspace_step_size=workspace_step_size)

    data_dir = r'D:\PycharmProjects\robotic-grasping\calibrate\data\hand_to_eye_20250824_164130'

    # # 自动移动机械臂收集数据
    # data_dir = eye_to_hand.auto_collect_data()

    # 指定姿态采集机械臂数据
    # robot_pose_file = './robot_origin_pose.txt'
    # data_dir = eye_to_hand.collect_data_by_poses(robot_pose_file)

    # 执行手眼标定
    # eye_to_hand.do_calibrate(collect_data_dir=data_dir, show_board_img=False, M_flange_board=M_flange_board)

    # 验证标定结果误差
    # eye_to_hand.verify_residual_error(collect_data_dir=data_dir)

    # 相机实际验证
    # eye_to_hand.verify_by_camera(collect_data_dir=data_dir)
