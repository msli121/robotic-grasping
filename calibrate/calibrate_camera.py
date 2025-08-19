# -*- coding: utf-8 -*-
# @Time       : 2025/8/16 17:39
# @File       : calibrate_camera.py
# @Description: 相机外参标定，通过SVD求解点集的刚性变换矩阵，完成相机坐标系到机械臂坐标系的变换矩阵camera2world求解
# 支持在线标定和离线标定
# 在线标定：通过机械臂移动到不同位置，采集相机图像和机械臂位姿数据，实时标定相机外参
# 离线标定：通过已经采集好的相机图像和机械臂位姿数据，离线标定相机外参
# @Author     : lms
# @Date       : 2025/8/16 17:39
import glob
import logging
import os
import time

import cv2
import numpy as np
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from scipy import optimize

from hardware.camera import RealSenseCamera
from robot.densor_robot import DensorRobot

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class CalibrateCamera:
    def __init__(self,
                 cam_id,
                 calib_grid_step,
                 checkerboard_offset_from_tool,
                 workspace_limits
                 ):
        self.calib_grid_step = calib_grid_step
        self.checkerboard_offset_from_tool = checkerboard_offset_from_tool

        # 机械臂在工作空间中的位置范围
        self.workspace_limits = workspace_limits
        # 相机
        self.camera = RealSenseCamera(device_id=cam_id)
        # 机械臂
        self.robot = DensorRobot()

        # 标定板中心在机械臂基坐标系下的三维坐标
        self.measured_pts = []
        # 标定板中心在相机坐标系下的三维坐标
        self.observed_pts = []
        # 标定板中心在相机图像上的二维像素坐标，用于结合相机内参计算相机坐标系下的值
        self.observed_pix = []
        # 相机外参，相机坐标系到机械臂基坐标系的变换矩阵
        self.camera2world = np.eye(4)

    @staticmethod
    def _get_rigid_transform(A, B):
        """
        Estimate rigid transform with SVD (from Nghia Ho)
        通过SVD计算刚性变换[R T] => B = R*A + T
        :param A: Nx3 matrix of points
        :param B: Nx3 matrix of points
        :return: R, t
        """
        assert len(A) == len(B)

        # 总点数
        N = A.shape[0]
        # 计算点集质心
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        # 点集去质心，让质心为原点
        AA = A - np.tile(centroid_A, (N, 1))
        BB = B - np.tile(centroid_B, (N, 1))
        # 构建协方差矩阵
        H = np.dot(np.transpose(AA), BB)
        # SVD 分解提取旋转矩阵
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        # 修正反射问题
        if np.linalg.det(R) < 0:  # Special reflection case
            Vt[2, :] *= -1
            R = np.dot(Vt.T, U.T)
        # 计算平移向量
        t = np.dot(-R, centroid_A.T) + centroid_B.T
        return R, t

    def _get_rigid_transform_error(self, z_scale=1):
        """
        Calculate the rigid transform RMS error
        计算修正后的相机点集和机械臂真实点集的刚性变换误差
        :param z_scale: 相机深度缩放因子
        :return RMS error
        """
        # 计算修正后的z坐标
        observed_z = np.squeeze(self.observed_pts[:, 2:] * z_scale)
        # 重新计算x坐标：(像素u - 主点x) * 修正后的z / 焦距x
        observed_x = np.multiply(np.squeeze(self.observed_pix[:, [0]]) - self.camera.intrinsics.ppx,
                                 observed_z / self.camera.intrinsics.fx)
        # 重新计算y坐标：(像素v - 主点y) * 修正后的z / 焦距y
        observed_y = np.multiply(np.squeeze(self.observed_pix[:, [1]]) - self.camera.intrinsics.ppy,
                                 observed_z / self.camera.intrinsics.fy)

        new_observed_pts = np.asarray([observed_x, observed_y, observed_z]).T

        # 用修正后的相机点集和机械臂真实点集，求解旋转矩阵R和平移向量t
        R, t = self._get_rigid_transform(np.asarray(new_observed_pts), np.asarray(self.measured_pts))
        # 构建4x4的相机到机械臂的变换矩阵camera2world
        t.shape = (3, 1)
        self.camera2world = np.concatenate((np.concatenate((R, t), axis=1), np.array([[0, 0, 0, 1]])), axis=0)

        # 将修正后的相机点通过R和t变换到机械臂坐标系，得到变换后的点
        registered_pts = np.dot(R, np.transpose(new_observed_pts)) + np.tile(t, (1, new_observed_pts.shape[0]))
        error = np.transpose(registered_pts) - self.measured_pts
        error = np.sum(np.multiply(error, error))
        rmse = np.sqrt(error / new_observed_pts.shape[0])
        return rmse

    def _generate_grid(self):
        """
        Construct 3D calibration grid across workspace
        生成机械臂工作空间中的3D网格点
        :return calibration grid points
        """
        # 将计算出的样本数量转换为整数
        x_num = int(np.ceil(1 + (self.workspace_limits[0][1] - self.workspace_limits[0][0]) / self.calib_grid_step))
        y_num = int(np.ceil(1 + (self.workspace_limits[1][1] - self.workspace_limits[1][0]) / self.calib_grid_step))
        z_num = int(np.ceil(1 + (self.workspace_limits[2][1] - self.workspace_limits[2][0]) / self.calib_grid_step))

        gridspace_x = np.linspace(self.workspace_limits[0][0], self.workspace_limits[0][1], x_num)
        gridspace_y = np.linspace(self.workspace_limits[1][0], self.workspace_limits[1][1], y_num)
        gridspace_z = np.linspace(self.workspace_limits[2][0], self.workspace_limits[2][1], z_num)

        calib_grid_x, calib_grid_y, calib_grid_z = np.meshgrid(gridspace_x, gridspace_y, gridspace_z)
        num_calib_grid_pts = calib_grid_x.shape[0] * calib_grid_x.shape[1] * calib_grid_x.shape[2]
        calib_grid_x.shape = (num_calib_grid_pts, 1)
        calib_grid_y.shape = (num_calib_grid_pts, 1)
        calib_grid_z.shape = (num_calib_grid_pts, 1)
        calib_grid_pts = np.concatenate((calib_grid_x, calib_grid_y, calib_grid_z), axis=1)
        return calib_grid_pts

    def _do_calibrate(self, data_save_dir=None):
        """
        执行标定
        """
        if data_save_dir is None:
            logger.error('请输入数据保存目录')
            return
        os.makedirs(data_save_dir, exist_ok=True)

        self.measured_pts = np.asarray(self.measured_pts)
        self.observed_pts = np.asarray(self.observed_pts)
        self.observed_pix = np.asarray(self.observed_pix)

        # 保证个数一致
        if len(self.measured_pts) != len(self.observed_pts) != len(self.observed_pix):
            logger.error('数据加载失败，点位信息数量不一致')
            return

        # 通过最小化误差来标定相机深度偏移
        logger.info('点位信息收集完毕，开始执行标定...')
        z_scale_init = 1
        optim_result = optimize.minimize(self._get_rigid_transform_error, np.asarray(z_scale_init),
                                         method='Nelder-Mead')
        camera_depth_offset = optim_result.x
        logger.info(f'标定结束，最优深度缩放系数为: {camera_depth_offset}')

        # 保存标定结果
        logger.info('开始保存结果...')
        camera_depth_scale_file = os.path.join(data_save_dir, f'camera_depth_scale.txt')
        logger.info(f'相机深度缩放系数 保存路径: {os.path.abspath(camera_depth_scale_file)}')
        np.savetxt(camera_depth_scale_file, camera_depth_offset, delimiter=' ')
        rmse = self._get_rigid_transform_error(z_scale=camera_depth_offset)
        logger.info(f'标定结果均方根误差(RMSE): {rmse}')
        camera_pose_file = os.path.join(data_save_dir, f'camera_pose.txt')
        np.savetxt(camera_pose_file, self.camera2world, delimiter=' ')
        logger.info(f'相机坐标系到机械臂坐标系变换矩阵 保存路径: {os.path.abspath(camera_pose_file)}')
        logger.info('标定完成！！！')

    def run(self):
        logging.info('开始标定...')

        # 计算空间坐标点
        calib_grid_pts = self._generate_grid()
        logger.info(f'工作空间总点数: {calib_grid_pts.shape[0]}')

        # 连接相机
        self.camera.connect()
        logger.info(f"相机连接成功...")
        K, dist = self.camera.get_K_and_dist()
        logger.info(f"相机内参: {K}")
        logger.info(f"相机畸变系数: {dist}")

        # 连接机器人
        logger.info(f"开始连接机器人...")
        self.robot.connect()
        home_position = [300.0, 5, 200.0, 127, 76, 122, 1]
        # 机器人移动到默认位置
        self.robot.send_position(home_position)
        # 等待机械臂到达指定位置
        time.sleep(2)

        # 标定照片保存的文件夹
        current_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
        data_save_dir = os.path.join(BASE_DIR, 'data', current_time)
        os.makedirs(data_save_dir, exist_ok=True)
        # # 标定结果保存路径
        # calibrate_result_dir = os.path.join(BASE_DIR, 'data')

        for index, tool_position in enumerate(calib_grid_pts):
            # 用tool_position替换home_position的前三个元素，并且乘以1000
            robot_position = tool_position * 1000
            robot_position = list(robot_position)
            robot_position.extend(home_position[3:])
            logger.info(f'位置{index:02d} 开始移动到指定位置: {robot_position}')

            # 机器人移动到指定位置
            self.robot.send_position(robot_position)
            # 等待机械臂到达指定位置
            time.sleep(2)
            logger.info(f'位置{index:02d} 移动到指定位置完成')

            # 寻找标定板中心坐标
            checkerboard_size = (8, 8)
            refine_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            image_bundle = self.camera.get_image_bundle()
            camera_color_img = image_bundle['rgb']
            camera_depth_img = image_bundle['aligned_depth']

            bgr_color_data = cv2.cvtColor(camera_color_img, cv2.COLOR_RGB2BGR)
            gray_data = cv2.cvtColor(bgr_color_data, cv2.COLOR_RGB2GRAY)
            checkerboard_found, corners = cv2.findChessboardCorners(gray_data, checkerboard_size, None,
                                                                    cv2.CALIB_CB_ADAPTIVE_THRESH)
            if checkerboard_found:
                corners_refined = cv2.cornerSubPix(gray_data, corners, checkerboard_size, (-1, -1), refine_criteria)
                # 显示角点图像
                bgr_color_img_copy = bgr_color_data.copy()
                cv2.drawChessboardCorners(bgr_color_img_copy, checkerboard_size, corners_refined,
                                          checkerboard_found)
                cv2.imshow("ImageWithCorners", bgr_color_img_copy)
                cv2.waitKey(1000)
                cv2.destroyAllWindows()
                # # 检查角点方向
                # if abs(corners[0][0][1] - corners[1][0][1]) > 10:
                #     logger.warning("角点识别方向与实际方向不一致，跳过处理")
                #     continue
                # 保存原图RGB
                img_origin_path = os.path.join(data_save_dir, f'{index:02d}_origin_rgb.png')
                cv2.imwrite(img_origin_path, bgr_color_data)
                # 保存带有角点的图像
                img_with_corner_path = os.path.join(data_save_dir, f'{index:02d}_corner_rgb.png')
                cv2.imwrite(img_with_corner_path, bgr_color_img_copy)
                # 获取标定板中心点的坐标
                center_point_left_up = np.round(corners_refined[27, 0, :]).astype(int)
                center_point_right_down = np.round(corners_refined[36, 0, :]).astype(int)
                checkerboard_pix = (center_point_left_up + center_point_right_down) // 2
                logger.info(f"位置{index:02d} 标定板中心点 像素坐标系: {checkerboard_pix}")

                # 使用cv2画出中心点
                color_copy = bgr_color_data.copy()
                cv2.circle(color_copy, checkerboard_pix, 3, (0, 0, 255), -1)
                # 显示图像
                cv2.imshow("ImageWithCenterPoint", color_copy)
                cv2.waitKey(1000)
                cv2.destroyAllWindows()
                # 保存带有中心点的图像
                img_with_center_point_path = os.path.join(data_save_dir, f'{index:02d}_center_rgb.png')
                cv2.imwrite(img_with_center_point_path, color_copy)
                # 保存深度图数据
                camera_depth_copy = camera_depth_img.copy()
                depth_data_path = os.path.join(data_save_dir, f'{index:02d}_depth_raw.npy')
                np.save(depth_data_path, camera_depth_copy)
                # 保存深度可视化图
                depth_visual_img_path = os.path.join(data_save_dir, f'{index:02d}_depth_visual.png')
                # 深度图预处理：归一化并转换为彩色图（便于可视化）
                depth_normalized = cv2.normalize(
                    camera_depth_copy.squeeze(),  # 移除单通道维度
                    None,
                    0, 255,
                    cv2.NORM_MINMAX,
                    dtype=cv2.CV_8U  # 转换为8位无符号整数
                )
                depth_vis = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)  # 应用彩色映射
                cv2.imwrite(depth_visual_img_path, depth_vis)
                # 获取并保存机器人位姿信息
                position = self.robot.get_current_position()
                robot_pose_path = os.path.join(data_save_dir, f'{index:02d}_robot_pose.txt')
                np.savetxt(robot_pose_path, position, delimiter=' ')

                # 像素坐标转相机坐标
                camera_z = camera_depth_img[checkerboard_pix[1]][checkerboard_pix[0]]
                camera_x = np.multiply(checkerboard_pix[0] - self.camera.intrinsics.ppx,
                                       camera_z / self.camera.intrinsics.fx)
                camera_y = np.multiply(checkerboard_pix[1] - self.camera.intrinsics.ppy,
                                       camera_z / self.camera.intrinsics.fy)
                if camera_z <= 0.05:
                    logger.error(f"位置{index:02d} 标定板中心点 相机深度值异常: {camera_z}")
                    continue

                # 保存像素坐标下的中心点坐标
                self.observed_pix.append(checkerboard_pix)
                logger.info(f"位置{index:02d} 标定板中心点 像素坐标系: {checkerboard_pix}")
                # 保存相机坐标系的中心点坐标
                camera_coord_center_point_position = np.array([camera_x, camera_y, camera_z])
                camera_coord_center_point_position = camera_coord_center_point_position.flatten()
                self.observed_pts.append(camera_coord_center_point_position)
                logger.info(f"位置{index:02d} 标定板中心点 相机坐标系: {camera_coord_center_point_position}")
                # 保存机械臂基坐标系下的中心点坐标
                robot_base_coord_center_point_position = tool_position + self.checkerboard_offset_from_tool
                self.measured_pts.append(robot_base_coord_center_point_position)
                logger.info(f"位置{index:02d} 标定板中心点 机械臂基坐标系: {robot_base_coord_center_point_position}")
            else:
                logger.error(f"位置{index:02d} 标定板未找到角点")
                continue

        # 机器人移动到默认位置
        self.robot.send_position(home_position)
        # 等待机械臂到达指定位置
        time.sleep(2)

        # 执行标定
        self._do_calibrate(data_save_dir=data_save_dir)

    def run_offline(self, data_save_dir=None, max_img_num=80):
        """
        使用已经拍摄好的rgb图、深度图、机械臂末端位姿坐标 离线标定
        """
        if data_save_dir is None:
            logger.error('请输入数据保存目录')
            return

        # 连接相机
        self.camera.connect()
        logger.info(f"相机连接成功...")
        K, dist = self.camera.get_K_and_dist()
        logger.info(f"相机内参: {K}")
        logger.info(f"相机畸变系数: {dist}")

        # 加载数据
        rgb_files = sorted(glob.glob(os.path.join(data_save_dir, '*origin_rgb.png')))
        depth_files = sorted(glob.glob(os.path.join(data_save_dir, '*depth_raw.npy')))
        robot_pose_files = sorted(glob.glob(os.path.join(data_save_dir, '*robot_pose.txt')))
        if len(rgb_files) != len(depth_files) != len(robot_pose_files):
            logger.error('数据加载失败，rgb图、深度图、机械臂位姿文件数量不一致')
            return
        # 文件排序
        rgb_files = sorted(rgb_files, key=lambda x: int(os.path.basename(x).split('_')[0]))
        depth_files = sorted(depth_files, key=lambda x: int(os.path.basename(x).split('_')[0]))
        robot_pose_files = sorted(robot_pose_files, key=lambda x: int(os.path.basename(x).split('_')[0]))

        checkerboard_size = (8, 8)
        refine_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # 加载数据
        handle_count = 0
        for rgb_file, depth_file, robot_pose_file in zip(rgb_files, depth_files, robot_pose_files):
            if handle_count > max_img_num:
                break
            index = int(os.path.basename(rgb_file).split('_')[0])
            bgr_color_img = cv2.imread(rgb_file)
            depth_img = np.load(depth_file)
            robot_pose = np.loadtxt(robot_pose_file, delimiter=' ')
            tool_position = robot_pose[:3] / 1000

            gray_data = cv2.cvtColor(bgr_color_img, cv2.COLOR_RGB2GRAY)
            checkerboard_found, corners = cv2.findChessboardCorners(gray_data, checkerboard_size, None,
                                                                    cv2.CALIB_CB_ADAPTIVE_THRESH)
            if checkerboard_found:
                corners_refined = cv2.cornerSubPix(gray_data, corners, checkerboard_size, (-1, -1), refine_criteria)
                # 显示角点图像
                bgr_color_img_copy = bgr_color_img.copy()
                cv2.drawChessboardCorners(bgr_color_img_copy, checkerboard_size, corners_refined,
                                          checkerboard_found)
                # cv2.imshow("ImageWithCorners", bgr_color_img_copy)
                # cv2.waitKey(1000)
                # cv2.destroyAllWindows()

                # 获取标定板中心点的坐标
                center_point_left_up = np.round(corners_refined[27, 0, :]).astype(int)
                center_point_right_down = np.round(corners_refined[36, 0, :]).astype(int)
                checkerboard_pix = (center_point_left_up + center_point_right_down) // 2
                logger.info(f"位置{index:02d} 标定板中心点 像素坐标系: {checkerboard_pix}")

                # 像素坐标转相机坐标
                camera_z = depth_img[checkerboard_pix[1]][checkerboard_pix[0]]
                camera_x = np.multiply(checkerboard_pix[0] - self.camera.intrinsics.ppx,
                                       camera_z / self.camera.intrinsics.fx)
                camera_y = np.multiply(checkerboard_pix[1] - self.camera.intrinsics.ppy,
                                       camera_z / self.camera.intrinsics.fy)
                if camera_z <= 0.05 or camera_z >= 0.7:
                    logger.error(f"位置{index:02d} 标定板中心点 相机深度值异常: {camera_z}")
                    continue

                # 保存像素坐标下的中心点坐标
                self.observed_pix.append(checkerboard_pix)
                logger.info(f"位置{index:02d} 标定板中心点 像素坐标系: {checkerboard_pix}")
                # 保存相机坐标系的中心点坐标
                camera_coord_center_point_position = np.array([camera_x, camera_y, camera_z])
                camera_coord_center_point_position = camera_coord_center_point_position.flatten()
                self.observed_pts.append(camera_coord_center_point_position)
                logger.info(f"位置{index:02d} 标定板中心点 相机坐标系: {camera_coord_center_point_position}")
                # 保存机械臂基坐标系下的中心点坐标
                robot_base_coord_center_point_position = tool_position + self.checkerboard_offset_from_tool
                self.measured_pts.append(robot_base_coord_center_point_position.flatten())
                logger.info(
                    f"位置{index:02d} 标定板中心点 机械臂基坐标系: {robot_base_coord_center_point_position}")

                handle_count += 1
            else:
                logger.error(f"位置{index:02d} 标定板中心点 未检测到")

        # 执行标定
        self._do_calibrate(data_save_dir=data_save_dir)

    def verify_calibration_by_realsense_camera(self, data_save_dir=None):
        """
        通过RealSense相机验证手眼标定结果，支持用户点击图像获取3D坐标
        优化点：增加错误处理、可视化标记、深度平滑、结果保存和用户提示
        """
        if not data_save_dir:
            logger.error(f"数据保存文件夹未指定")
            return
        # ========== 配置与常量定义 ==========
        camera2robot = np.loadtxt(os.path.join(data_save_dir, 'camera_pose.txt'), delimiter=' ')
        cam_depth_scale = np.loadtxt(os.path.join(data_save_dir, 'camera_depth_scale.txt'), delimiter=' ')

        SAVE_RESULTS = True  # 是否保存测量结果
        RESULTS_FILE = os.path.join(data_save_dir, "calibration_verification_results.txt")

        # 存储测量结果
        measurement_results = []

        # ========== 初始化相机（带错误处理） ==========
        try:
            self.camera.connect()
            # 获取相机内参（优先使用实时获取的内参）
            # 获取相机内参
            fx = self.camera.intrinsics.fx
            fy = self.camera.intrinsics.fy
            cx = self.camera.intrinsics.ppx
            cy = self.camera.intrinsics.ppy
            print(f"相机连接成功 - 内参: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
        except Exception as e:
            print(f"相机初始化失败: {str(e)}")
            return

        # ========== 鼠标回调函数（增强版） ==========
        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # 获取参数
                rgb = param['rgb']
                depth = param['aligned_depth']
                height, width = rgb.shape[:2]

                # 检查点击位置是否在图像范围内
                if not (0 <= x < width and 0 <= y < height):
                    print("点击位置超出图像范围")
                    return

                # 从对齐的深度图获取深度值（注意坐标顺序）
                depth_value = depth[y, x] * cam_depth_scale
                depth_value = depth_value[0]
                if depth_value < 0.1 or depth_value > 0.7:  # 合理深度范围判断
                    print(f"深度值({depth_value:.3f}m)超出有效范围(0.1-0.7m)")
                    return

                # 2. 将像素点投影到相机坐标系
                Xc = (x - cx) * depth_value / fx
                Yc = (y - cy) * depth_value / fy
                Zc = depth_value
                camera_coord = np.array([Xc, Yc, Zc])

                # 转换为齐次坐标
                camera_coord_homog = np.append(camera_coord, [1]).reshape(4, 1)

                # 转换到机器人基坐标系
                robot_coord = np.dot(camera2robot, camera_coord_homog)
                Pbase = robot_coord[:3].flatten()  # 移除齐次坐标

                # 4. 显示和记录结果
                result_str = (f"像素点: ({x},{y}) → 深度: {depth_value:.3f}m → "
                              f"相机坐标: X={Xc:.4f}m, Y={Yc:.4f}m, Z={Zc:.4f}m → "
                              f"基座坐标: X={Pbase[0]:.4f}m, Y={Pbase[1]:.4f}m, Z={Pbase[2]:.4f}m")
                print(result_str)

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
            print("\n操作说明:")
            print("1. 点击图像上的点获取其在机械臂基座坐标系中的坐标")
            print("2. 按 's' 保存当前测量结果到文件")
            print("3. 按 'r' 清除所有测量结果")
            print("4. 按 'ESC' 退出程序")

            while True:
                # 获取图像
                images = self.camera.get_image_bundle()
                if not images or 'rgb' not in images or 'aligned_depth' not in images:
                    print("获取图像失败，重试...")
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
                    print("程序退出")
                    break
                elif key == ord('s') and SAVE_RESULTS:  # 保存结果
                    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
                        for i, res in enumerate(measurement_results, 1):
                            f.write(f"测量点 {i}:\n")
                            f.write(f"  像素坐标: {res['pixel']}\n")
                            f.write(f"  深度值: {res['depth']:.4f}m\n")
                            f.write(f"  相机坐标: {res['camera_coords']}\n")
                            f.write(f"  基座坐标: {res['base_coords']}\n\n")
                    print(f"已保存 {len(measurement_results)} 个测量结果到 {RESULTS_FILE}")
                elif key == ord('r'):  # 清除结果
                    measurement_results.clear()
                    print("已清除所有测量结果")

        except Exception as e:
            print(f"程序运行出错: {str(e)}")
        finally:
            # 资源清理
            cv2.destroyAllWindows()
            print("资源已释放")

            # 自动保存结果（如果有）
            if SAVE_RESULTS and measurement_results:
                with open(RESULTS_FILE, 'w') as f:
                    for i, res in enumerate(measurement_results, 1):
                        f.write(f"测量点 {i}:\n")
                        f.write(f"  像素坐标: {res['pixel']}\n")
                        f.write(f"  深度值: {res['depth']:.4f}m\n")
                        f.write(f"  相机坐标: {res['camera_coords']}\n")
                        f.write(f"  基座坐标: {res['base_coords']}\n\n")
                print(f"自动保存 {len(measurement_results)} 个测量结果到 {RESULTS_FILE}")


if __name__ == '__main__':
    cam_id = 246422072474
    checkerboard_offset_from_tool = [0.065, 0.0, 0.0],
    workspace_limits = np.asarray([[0.30, 0.42], [-0.10, 0.10], [0.02, 0.15]])
    calib_grid_step = 0.05
    calibrate_camera = CalibrateCamera(cam_id=cam_id,
                                       calib_grid_step=calib_grid_step,
                                       checkerboard_offset_from_tool=checkerboard_offset_from_tool,
                                       workspace_limits=workspace_limits)
    # calibrate_camera.run()
    data_save_dir = r'D:\PycharmProjects\robotic-grasping\calibrate\data\20250819001632'
    calibrate_camera.run_offline(data_save_dir=data_save_dir)

    calibrate_camera.verify_calibration_by_realsense_camera(data_save_dir=data_save_dir)
