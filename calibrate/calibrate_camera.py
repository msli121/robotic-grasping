# -*- coding: utf-8 -*-
# @Time       : 2025/8/16 17:39
# @File       : calibrate_camera.py
# @Description: 相机外参标定，通过SVD求解点集的刚性变换矩阵，完成相机坐标系到机械臂坐标系的变换矩阵camera2world求解
# @Author     : lms
# @Date       : 2025/8/16 17:39

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
        gridspace_x = np.linspace(self.workspace_limits[0][0], self.workspace_limits[0][1],
                                  np.ceil(1 + (self.workspace_limits[0][1] - self.workspace_limits[0][
                                      0]) / self.calib_grid_step))
        gridspace_y = np.linspace(self.workspace_limits[1][0], self.workspace_limits[1][1],
                                  np.ceil(1 + (self.workspace_limits[1][1] - self.workspace_limits[1][
                                      0]) / self.calib_grid_step))
        gridspace_z = np.linspace(self.workspace_limits[2][0], self.workspace_limits[2][1],
                                  np.ceil(1 + (self.workspace_limits[2][1] - self.workspace_limits[2][
                                      0]) / self.calib_grid_step))
        calib_grid_x, calib_grid_y, calib_grid_z = np.meshgrid(gridspace_x, gridspace_y, gridspace_z)
        num_calib_grid_pts = calib_grid_x.shape[0] * calib_grid_x.shape[1] * calib_grid_x.shape[2]
        calib_grid_x.shape = (num_calib_grid_pts, 1)
        calib_grid_y.shape = (num_calib_grid_pts, 1)
        calib_grid_z.shape = (num_calib_grid_pts, 1)
        calib_grid_pts = np.concatenate((calib_grid_x, calib_grid_y, calib_grid_z), axis=1)
        return calib_grid_pts

    def run(self):
        logging.info('开始标定...')
        # 连接相机
        self.camera.connect()
        logger.info(f"相机连接成功...")
        K, dist = self.camera.get_K_and_dist()
        logger.info(f"相机内参: {K}")
        logger.info(f"相机畸变系数: {dist}")

        # 连接机器人
        self.robot.connect()
        home_position = [250.0, 0.0, 240.0, -140, -75, -41, 9]
        # 机器人移动到默认位置
        self.robot.send_position(home_position)
        # 等待机械臂到达指定位置
        time.sleep(2)

        calib_grid_pts = self._generate_grid()
        logger.info(f'工作空间总点数: {calib_grid_pts.shape[0]}')

        # 标定照片保存的文件夹
        image_save_dir = os.path.join(BASE_DIR, 'data', 'calibration_images')
        os.makedirs(image_save_dir, exist_ok=True)
        # 标定结果保存路径
        calibrate_result_dir = os.path.join(BASE_DIR, 'data')

        for index, tool_position in enumerate(calib_grid_pts):
            # 用tool_position替换home_position的前三个元素，并且乘以1000
            robot_position = tool_position * 1000
            robot_position = list(robot_position)
            robot_position.extend(home_position[3:])
            logger.info(f'位置{index} 开始移动到指定位置: {robot_position}')

            # 机器人移动到指定位置
            self.robot.send_position(robot_position)
            # 等待机械臂到达指定位置
            time.sleep(2)
            logger.info(f'位置{index} 移动到指定位置完成')

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
                # 保存带有角点的图像
                img_with_corner_path = os.path.join(image_save_dir, f'{index}_img_with_corner.png')
                cv2.imwrite(img_with_corner_path, bgr_color_img_copy)

                # 获取标定板中心点的坐标
                center_point_left_up = np.round(corners_refined[27, 0, :]).astype(int)
                center_point_right_down = np.round(corners_refined[36, 0, :]).astype(int)
                checkerboard_pix = (center_point_left_up + center_point_right_down) // 2
                logger.info(f"位置{index} 标定板中心点 像素坐标系: {checkerboard_pix}")

                # 使用cv2画出中心点
                color_copy = bgr_color_data.copy()
                cv2.circle(color_copy, checkerboard_pix, 3, (0, 0, 255), -1)
                # 显示图像
                cv2.imshow("ImageWithCenterPoint", color_copy)
                cv2.waitKey(1000)
                cv2.destroyAllWindows()
                # 保存带有中心点的图像
                img_with_center_point_path = os.path.join(image_save_dir, f'{index}_img_with_center_point.png')
                cv2.imwrite(img_with_center_point_path, color_copy)

                # 像素坐标转相机坐标
                camera_z = camera_depth_img[checkerboard_pix[1]][checkerboard_pix[0]]
                camera_x = np.multiply(checkerboard_pix[0] - self.camera.intrinsics.ppx,
                                       camera_z / self.camera.intrinsics.fx)
                camera_y = np.multiply(checkerboard_pix[1] - self.camera.intrinsics.ppy,
                                       camera_z / self.camera.intrinsics.fy)
                if camera_z <= 0.05:
                    logger.error(f"位置{index} 标定板中心点 相机深度值异常: {camera_z}")
                    continue

                # 保存像素坐标下的中心点坐标
                self.observed_pix.append(checkerboard_pix)
                logger.info(f"位置{index} 标定板中心点 像素坐标系: {checkerboard_pix}")
                # 保存相机坐标系的中心点坐标
                camera_coord_center_point_position = np.array([camera_x, camera_y, camera_z])
                camera_coord_center_point_position = camera_coord_center_point_position.flatten()
                self.observed_pts.append(camera_coord_center_point_position)
                logger.info(f"位置{index} 标定板中心点 相机坐标系: {camera_coord_center_point_position}")
                # 保存机械臂基坐标系下的中心点坐标
                robot_base_coord_center_point_position = tool_position + self.checkerboard_offset_from_tool
                self.measured_pts.append(robot_base_coord_center_point_position)
                logger.info(f"位置{index} 标定板中心点 机械臂基坐标系: {robot_base_coord_center_point_position}")
            else:
                logger.error(f"位置{index} 标定板未找到角点")
                continue

        self.measured_pts = np.asarray(self.measured_pts)
        self.observed_pts = np.asarray(self.observed_pts)
        self.observed_pix = np.asarray(self.observed_pix)

        # 通过最小化误差来标定相机深度偏移
        logger.info('开始标定...')
        z_scale_init = 1
        optim_result = optimize.minimize(self._get_rigid_transform_error, np.asarray(z_scale_init),
                                         method='Nelder-Mead')
        camera_depth_offset = optim_result.x
        logger.info(f'标定结束，最优深度缩放系数为: {camera_depth_offset}')

        # 保存标定结果
        logger.info('开始保存结果...')
        current_time_str = time.strftime('%Y%m%d%H%M%S', time.localtime())
        camera_depth_scale_file = os.path.join(calibrate_result_dir, f'camera_depth_scale_{current_time_str}.txt')
        logger.info(f'相机深度缩放系数 保存路径: {os.path.abspath(camera_depth_scale_file)}')
        np.savetxt(camera_depth_scale_file, camera_depth_offset, delimiter=' ')
        rmse = self._get_rigid_transform_error(camera_depth_offset)
        logger.info(f'标定结果均方根误差(RMSE): {rmse}')
        camera_pose_file = os.path.join(calibrate_result_dir, f'camera_pose_{current_time_str}.txt')
        np.savetxt(camera_pose_file, self.camera2world, delimiter=' ')
        logger.info(f'相机坐标系到机械臂坐标系变换矩阵 保存路径: {os.path.abspath(camera_pose_file)}')
        logger.info('标定完成！！！')


if __name__ == '__main__':
    cam_id = 246422072474,
    calib_grid_step = 0.03
    # 标定板中心到夹具中心的偏移
    checkerboard_offset_from_tool = [0.018 * 4, 0.0, 0.0],
    workspace_limits = np.asarray([[0.25, 0.30], [-0.10, 0.10], [0.05, 0.15]])
    calibrate_camera = CalibrateCamera(cam_id=cam_id,
                                       calib_grid_step=calib_grid_step,
                                       checkerboard_offset_from_tool=checkerboard_offset_from_tool,
                                       workspace_limits=workspace_limits)
    calibrate_camera.run()
