#!/usr/bin/env python
import logging
import os
import time

import cv2
import numpy as np
from scipy import optimize

from hardware.camera import RealSenseCamera
from robot.densor_robot import DensorRobot


# This import registers the 3D projection, but is otherwise unused.
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


class MyCalibration:
    def __init__(self,
                 cam_id,
                 calib_grid_step,
                 checkerboard_offset_from_tool,
                 workspace_limits
                 ):
        self.calib_grid_step = calib_grid_step
        self.checkerboard_offset_from_tool = checkerboard_offset_from_tool

        # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
        self.workspace_limits = workspace_limits

        self.camera = RealSenseCamera(device_id=cam_id)

        self.measured_pts = []
        self.observed_pts = []
        self.observed_pix = []
        self.camera2world = np.eye(4)

        homedir = os.path.join(os.path.expanduser('~'), "grasp-comms")
        os.makedirs(homedir, exist_ok=True)
        self.move_completed = os.path.join(homedir, "move_completed.npy")
        self.tool_position = os.path.join(homedir, "tool_position.npy")

    @staticmethod
    def _get_rigid_transform(A, B):
        """
        Estimate rigid transform with SVD (from Nghia Ho)
        """
        assert len(A) == len(B)

        N = A.shape[0]  # Total points
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        AA = A - np.tile(centroid_A, (N, 1))  # Centre the points
        BB = B - np.tile(centroid_B, (N, 1))
        H = np.dot(np.transpose(AA), BB)  # Dot is matrix multiplication for array
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        if np.linalg.det(R) < 0:  # Special reflection case
            Vt[2, :] *= -1
            R = np.dot(Vt.T, U.T)
        t = np.dot(-R, centroid_A.T) + centroid_B.T
        return R, t

    def _get_rigid_transform_error(self, z_scale):
        """
        Calculate the rigid transform RMS error

        :return RMS error
        """
        # Apply z offset and compute new observed points using camera intrinsics
        observed_z = np.squeeze(self.observed_pts[:, 2:] * z_scale)
        observed_x = np.multiply(np.squeeze(self.observed_pix[:, [0]]) - self.camera.intrinsics.ppx,
                                 observed_z / self.camera.intrinsics.fx)
        observed_y = np.multiply(np.squeeze(self.observed_pix[:, [1]]) - self.camera.intrinsics.ppy,
                                 observed_z / self.camera.intrinsics.fy)

        new_observed_pts = np.asarray([observed_x, observed_y, observed_z]).T

        # Estimate rigid transform between new observed points and measured points
        R, t = self._get_rigid_transform(np.asarray(new_observed_pts), np.asarray(self.measured_pts))
        t.shape = (3, 1)
        self.camera2world = np.concatenate((np.concatenate((R, t), axis=1), np.array([[0, 0, 0, 1]])), axis=0)

        # Compute rigid transform error
        registered_pts = np.dot(R, np.transpose(new_observed_pts)) + np.tile(t, (1, new_observed_pts.shape[0]))
        error = np.transpose(registered_pts) - self.measured_pts
        error = np.sum(np.multiply(error, error))
        rmse = np.sqrt(error / new_observed_pts.shape[0])
        return rmse

    def _generate_grid(self):
        """
        Construct 3D calibration grid across workspace

        :return calibration grid points
        """
        gridspace_x = np.linspace(self.workspace_limits[0][0], self.workspace_limits[0][1],
                                  int(1 + (self.workspace_limits[0][1] - self.workspace_limits[0][
                                      0]) / self.calib_grid_step))
        gridspace_y = np.linspace(self.workspace_limits[1][0], self.workspace_limits[1][1],
                                  int(1 + (self.workspace_limits[1][1] - self.workspace_limits[1][
                                      0]) / self.calib_grid_step))
        gridspace_z = np.linspace(self.workspace_limits[2][0], self.workspace_limits[2][1],
                                  int(1 + (self.workspace_limits[2][1] - self.workspace_limits[2][
                                      0]) / self.calib_grid_step))
        calib_grid_x, calib_grid_y, calib_grid_z = np.meshgrid(gridspace_x, gridspace_y, gridspace_z)
        num_calib_grid_pts = calib_grid_x.shape[0] * calib_grid_x.shape[1] * calib_grid_x.shape[2]
        calib_grid_x.shape = (num_calib_grid_pts, 1)
        calib_grid_y.shape = (num_calib_grid_pts, 1)
        calib_grid_z.shape = (num_calib_grid_pts, 1)
        calib_grid_pts = np.concatenate((calib_grid_x, calib_grid_y, calib_grid_z), axis=1)
        return calib_grid_pts

    def init(self):
        # Connect to camera
        self.camera.connect()
        logging.info(self.camera.intrinsics)

        # 机器人作为server端
        robot_ip = "192.168.1.11"
        robot_port = 5002
        self.robot = DensorRobot(host=robot_ip, port=robot_port)  # 传入机器人的IP地址和端口号
        self.home_position = [250.0, 0.0, 240.0, -140, -75, -41, 9]
        # 机器人移动到默认位置
        self.robot.send_position(self.home_position)
        # 等待机械臂到达指定位置
        time.sleep(2)

    def run(self):
        calib_grid_pts = self._generate_grid()
        logging.info(f'标定位置总个数: {calib_grid_pts.shape[0]}')
        # 标定照片保存的文件夹
        image_save_dir = os.path.join('saved_data', 'calibration_images')
        os.makedirs(image_save_dir, exist_ok=True)
        try:
            for index, tool_position in enumerate(calib_grid_pts):
                # 用tool_position替换home_position的前三个元素，并且乘以1000
                robot_position = tool_position * 1000
                robot_position = list(robot_position)
                robot_position.extend(self.home_position[3:])
                logging.info(f'位置{index} 移动到指定位置: {robot_position}')

                # 机器人移动到指定位置
                self.robot.send_position(robot_position)
                # 等待机械臂到达指定位置
                time.sleep(2)

                # 寻找标定版角度坐标
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
                    cv2.imshow("RGB", bgr_color_img_copy)
                    cv2.waitKey(1000)
                    cv2.destroyAllWindows()
                    # 检查角点方向
                    if abs(corners[0][0][1] - corners[1][0][1]) > 10:
                        logging.warning("角点识别方向与实际方向不一致，跳过处理")
                        continue
                    # 保存带有角点的图像
                    img_with_corner_path = os.path.join(image_save_dir, f'{index}_img_with_corner.png')
                    cv2.imwrite(img_with_corner_path, bgr_color_img_copy)

                    # 获取标定板中心点的坐标
                    center_point_left_up = np.round(corners_refined[27, 0, :]).astype(int)
                    center_point_right_down = np.round(corners_refined[36, 0, :]).astype(int)
                    checkerboard_pix = (center_point_left_up + center_point_right_down) // 2
                    logging.info(f"位置{index} 标定板中心点 像素坐标系: {checkerboard_pix}")

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
                    checkerboard_z = camera_depth_img[checkerboard_pix[1]][checkerboard_pix[0]]
                    checkerboard_x = np.multiply(checkerboard_pix[0] - self.camera.intrinsics.ppx,
                                                 checkerboard_z / self.camera.intrinsics.fx)
                    checkerboard_y = np.multiply(checkerboard_pix[1] - self.camera.intrinsics.ppy,
                                                 checkerboard_z / self.camera.intrinsics.fy)
                    if checkerboard_z == 0:
                        continue

                    # 保存像素坐标下的中心点坐标
                    self.observed_pix.append(checkerboard_pix)
                    # 保存相机坐标系的中心点坐标
                    camera_coord_center_point_position = np.array([checkerboard_x, checkerboard_y, checkerboard_z])
                    camera_coord_center_point_position = camera_coord_center_point_position.flatten()
                    logging.info(f"位置{index} 标定板中心点 相机坐标系: {camera_coord_center_point_position}")
                    self.observed_pts.append(camera_coord_center_point_position)
                    # 保存机械臂基坐标系下的中心点坐标
                    robot_base_coord_center_point_position = tool_position + self.checkerboard_offset_from_tool
                    logging.info(f"位置{index} 标定板中心点 机械臂基坐标系: {robot_base_coord_center_point_position}")
                    self.measured_pts.append(robot_base_coord_center_point_position)
                else:
                    logging.error('Checker board not found')

            self.measured_pts = np.asarray(self.measured_pts)
            self.observed_pts = np.asarray(self.observed_pts)
            self.observed_pix = np.asarray(self.observed_pix)

            # Optimize z scale w.r.t. rigid transform error
            logging.info('Calibrating...')
            z_scale_init = 1
            optim_result = optimize.minimize(self._get_rigid_transform_error, np.asarray(z_scale_init),
                                             method='Nelder-Mead')
            camera_depth_offset = optim_result.x

            # Save camera optimized offset and camera pose
            logging.info('Saving...')
            np.savetxt('saved_data/camera_depth_scale.txt', camera_depth_offset, delimiter=' ')
            np.savetxt('saved_data/camera_pose.txt', self.camera2world, delimiter=' ')
            rmse = self._get_rigid_transform_error(camera_depth_offset)
            logging.info(f'均方误差: {rmse}')
            np.savetxt('saved_data/rmse.txt', [rmse], delimiter=' ')
            logging.info('手眼标定完成.')
            # 机器人移动到默认位置
            self.robot.send_position(self.home_position)
            # 等待机械臂到达指定位置
            time.sleep(2)
        except Exception as e:
            logging.error(f'手眼标定失败: {e}')
        finally:
            # 机器人移动到默认位置
            self.robot.send_position(self.home_position)
            # 等待机械臂到达指定位置
            time.sleep(2)
            self.robot.close()


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    calibration = MyCalibration(
        cam_id=246422072474,
        calib_grid_step=0.03,
        checkerboard_offset_from_tool=[0.175, 0.0, 0.0],
        workspace_limits=np.asarray([[0.25, 0.30], [-0.10, 0.10], [0.05, 0.15]])
    )
    calibration.init()
    #calibration.run()
