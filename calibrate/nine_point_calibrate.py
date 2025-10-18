import logging
import os
import time

import colorlog
import cv2
import numpy as np
import pyrealsense2 as rs
from scipy.optimize import least_squares

from calibrate.utils import normalize_corner_order, robot_pose_to_homogeneous_matrix
from hardware.camera import RealSenseCamera
from robot.densor_robot import DensorRobot
from robot.gripper_wifi_controller import GripperWiFiController


# ==============================================================================
#                              配置日志 (保持不变)
# ==============================================================================
def setup_logger():
    logger = logging.getLogger(__name__)
    if logger.handlers: return logger
    logger.propagate = False
    log_colors = {'DEBUG': 'cyan', 'INFO': 'green', 'WARNING': 'yellow', 'ERROR': 'red', 'CRITICAL': 'bold_red'}
    formatter = colorlog.ColoredFormatter(fmt='%(log_color)s%(asctime)s - %(levelname)s - %(message)s',
                                          log_colors=log_colors)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    return logger


logger = setup_logger()
np.set_printoptions(precision=8, suppress=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def map_x_to_z(x):
    """
    将x值从[110, 230]范围映射到z值[-8, -38]范围，x增大时z减小

    参数:
        x: 输入值，应在[110, 230]范围内

    返回:
        对应的z值，在[-8, -38]范围内

    异常:
        ValueError: 当x不在[110, 230]范围内时抛出
    """
    # 定义x和z的范围
    min_x, max_x = 110, 230
    min_z, max_z = -38, -8  # 注意z的最小值和最大值

    # 检查x是否在有效范围内
    if x >= max_x:
        return min_z
    if x <= min_x:
        return max_z

    # 计算线性映射，x增大时z减小
    # 公式：z = max_z - (x - min_x) * (max_z - min_z) / (max_x - min_x)
    z = max_z - (x - min_x) * (max_z - min_z) / (max_x - min_x)

    return z


# TCP 四点标定
# 强烈建议的TCP标定步骤：
# 物理加固 (尽最大努力):
# 加重基座: 在您的矿泉水瓶周围，再多压几本厚重的书或者其他重物，让它的基座尽可能地稳固。
# 缩短笔杆: 如果可能，使用一支尽可能短的、粗壮的圆珠笔，或者只用笔芯的一部分，来减小弹性形变。
# 粘牢: 使用更多、更宽的胶带，将基座牢牢地粘在地面上。
# 执行“对称性四点法” (操作核心):
# 点1 vs 点2 (前后对称):
# 点1: 从正前方，以一个俯仰姿态，向后轻柔触碰。
# 点2: 从正后方，以相似的俯仰姿态，向前轻柔触碰。
# 目的：抵消前后方向的弹性形变和基座移动。
# 点3 vs 点4 (左右对称 + 强力自转):
# 点3: 从正左方，以一个侧摆姿态，并J6自转+90度，向右轻柔触碰。
# 点4: 从正右方，以一个侧摆姿态，并J6自转-90度，向左轻柔触碰。
# 目的：抵消左右方向的弹性形变和基座移动，同时强力约束XY平面的计算。
# 触碰原则:
# 极其轻柔: 您的目标是“刚刚碰到”，而不是“施加压力”。在示教器的低速模式下，以最小的步进单位去逼近。
# 保持一致: 尽量保证每次触碰时，您感觉到的接触力道都是一致的。


# 如何验证TCP标定的可靠性
# 方法一：定点旋转测试 (定性观察)
# 这是最快速、最直观的验证方法。
# 激活新工具: 在示教器上，确保您已经激活了刚刚标定好的工具坐标系（例如TOOL 2）。
# 定位到参考点: 将机械臂的TCP（即探针尖端）移动到空间中的任意一个固定参考点（不一定需要是标定时用的那个点，但同样需要尖锐且固定）。
# 切换到“工具坐标系”模式: 在示教器的手动操作模式中，将运动模式从“基座坐标系”或“关节坐标系”切换到**“工具坐标系 (Tool Frame)”**。
# 执行旋转:
# 现在，只操作示教器上的旋转指令（例如，Rx+, Rx-, Ry+, Ry-）。
# 观察现象:
# 如果TCP标定准确: 您会看到机械臂的臂身（特别是第4、5、6轴）在剧烈运动，但探针的尖端几乎保持在原地纹丝不动，就像一个圆心。
# 如果TCP标定不准: 您会看到探针的尖端在空中画出一个明显的小圆弧或小轨迹。这个圆弧的半径大小，就直观地反映了您TCP标定的误差大小。
# 结论: 如果笔尖基本不动，恭喜您，TCP标定非常成功。如果画圆明显，请放弃这个TCP，回到第一部分，重新进行一次更精确的标定。

# 标定板摆放
# 将棋盘格倾斜摆放时，例如给它一个20-40度的倾斜角（相对于相机的Z轴），产生Z轴上一定的高度差

# ==============================================================================
#                        核心计算与验证函数 (保持不变)
# ==============================================================================
def rigid_transform(src_pts, dst_pts, calc_scale=False):
    """Calculates the optimal rigid transform from src_pts to dst_pts.

    The returned transform minimizes the following least-squares problem
        r = dst_pts - (R @ src_pts + t)
        s = sum(r**2))

    If calc_scale is True, the similarity transform is solved, with the residual being
        r = dst_pts - (scale * R @ src_pts + t)
    where scale is a scalar.

    Parameters
    ----------
    src_pts: matrix of points stored as rows (e.g. Nx3)
    dst_pts: matrix of points stored as rows (e.g. Nx3)
    calc_scale: if True solve for scale

    Returns
    -------
    R: rotation matrix
    t: translation column vector
    scale: scalar, scale=1.0 if calc_scale=False
    """

    dim = src_pts.shape[1]

    if src_pts.shape != dst_pts.shape:
        raise ValueError(
            f"src and dst points aren't the same matrix size {src_pts.shape=} != {dst_pts.shape=}"
        )

    if not (dim == 2 or dim == 3):
        raise ValueError(f"Points must be 2D or 3D, src_pts.shape[1] = {dim}")

    if src_pts.shape[0] < dim:
        raise ValueError(f"Not enough points, expect >= {dim} points")

    # find mean/centroid
    centroid_src = np.mean(src_pts, axis=0)
    centroid_dst = np.mean(dst_pts, axis=0)

    centroid_src = centroid_src.reshape(-1, dim)
    centroid_dst = centroid_dst.reshape(-1, dim)

    # subtract mean
    # NOTE: doing src_pts -= centroid_src will modifiy input!
    src_pts = src_pts - centroid_src
    dst_pts = dst_pts - centroid_dst

    # the cross-covariance matrix minus the mean calculation for each element
    # https://en.wikipedia.org/wiki/Cross-covariance_matrix
    H = src_pts.T @ dst_pts

    rank = np.linalg.matrix_rank(H)

    if dim == 2 and rank == 0:
        raise ValueError(
            f"Insufficent matrix rank. For 2D points expect rank >= 1 but got {rank}. Maybe your points are all the same?"
        )
    elif dim == 3 and rank <= 1:
        raise ValueError(
            f"Insufficent matrix rank. For 3D points expect rank >= 2 but got {rank}. Maybe your points are collinear?"
        )

    # find rotation
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    # https://en.wikipedia.org/wiki/Kabsch_algorithm
    det = np.linalg.det(R)
    if det < 0:
        print(f"det(R) = {det}, reflection detected!, correcting for it ...")
        S = np.eye(dim)
        S[-1, -1] = -1
        R = Vt.T @ S @ U.T

    if calc_scale:
        scale = np.sqrt(np.mean(dst_pts ** 2) / np.mean(src_pts ** 2))
    else:
        scale = 1.0

    t = -scale * R @ centroid_src.T + centroid_dst.T

    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t.flatten()
    return M


def verify_transformation(M, points_A, points_B):
    errors = []
    for i in range(points_A.shape[0]):
        p_A_hom = np.append(points_A[i], 1)
        p_B_predicted = (M @ p_A_hom)[:3]
        p_B_actual = points_B[i]
        error = np.linalg.norm(p_B_predicted - p_B_actual)
        errors.append(error)
        print(
            f"点 {i + 1}: 实际 P_base = {p_B_actual}, 预测 P_base = {p_B_predicted}, 误差 = {error * 1000:.4f} mm")
    errors = np.array(errors)
    print("\n" + "=" * 20 + " 验证结果总结 " + "=" * 20)
    print(f"平均重投影误差: {np.mean(errors) * 1000:.4f} mm")
    print(f"最大重投影误差: {np.max(errors) * 1000:.4f} mm")
    print(f"误差标准差: {np.std(errors) * 1000:.4f} mm")
    print(f"RMSE 误差: {np.sqrt(np.mean(errors ** 2)) * 1000:.4f} mm")


class CameraDataCollector:
    def __init__(self, config):
        self.config = config
        self.pipeline = None
        self.save_dir = self.config['save_dir']
        self.intrinsics = None
        self.cam_matrix = None
        self.dist_coeffs = None
        os.makedirs(self.save_dir, exist_ok=True)

    def connect_camera(self):
        """连接相机并获取内参，只执行一次"""
        if self.pipeline:  # 如果已经连接，则直接返回
            return True
        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(str(self.config['camera_id']))
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            profile = self.pipeline.start(config)

            # 获取内参并保存
            color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
            self.intrinsics = color_profile.get_intrinsics()
            self.cam_matrix = np.array([[self.intrinsics.fx, 0, self.intrinsics.ppx],
                                        [0, self.intrinsics.fy, self.intrinsics.ppy],
                                        [0, 0, 1]])
            self.dist_coeffs = np.array(self.intrinsics.coeffs)

            print("Realsense相机连接成功。")
            print(f"使用相机内参: \n{self.cam_matrix}")
            print(f"使用相机畸变系数: {self.dist_coeffs}")
            return True
        except Exception as e:
            logger.error(f"连接Realsense相机失败: {e}")
            return False

    def disconnect_camera(self):
        """断开相机连接"""
        if self.pipeline:
            self.pipeline.stop()
            self.pipeline = None
            print("相机已关闭。")

    def collect_by_solvepnp(self):
        """
        【方法一】: 使用solvePnP整体解算棋盘格位姿，计算所有角点的3D坐标。
        这是理论上更精确、更稳健的方法。
        """
        print("\n" + "=" * 20 + " 开始使用 solvePnP 方法采集数据 " + "=" * 20)
        if not self.connect_camera():
            return

        print("请确保棋盘格在相机视野内清晰可见，且机器人已移开。")
        print("3秒后将自动捕获图像...")
        time.sleep(3)

        try:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                logger.error("未能获取彩色图像帧。")
                return

            color_image = np.asanyarray(color_frame.get_data())
            gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            chessboard_size = self.config['chessboard_size']
            chessboard_grid_size = self.config['chessboard_grid_size']
            objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * chessboard_grid_size

            ret, corners = cv2.findChessboardCorners(gray_image, chessboard_size, None)
            if not ret:
                logger.error("在图像中未能找到棋盘格角点！请检查相机位置和光照。")
                return

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray_image, corners, (11, 11), (-1, -1), criteria)

            # 角度归一化
            corners = normalize_corner_order(corners, chessboard_size)
            # 显示角点
            cv2.drawChessboardCorners(color_image, chessboard_size, corners, ret)
            cv2.imshow("Detected Corners", color_image)
            cv2.waitKey(3000)
            cv2.destroyAllWindows()
            # 保存角点
            cv2.imwrite(os.path.join(self.save_dir, "solvePnP_corners_detection.png"), color_image)

            # 计算相机位姿
            ret_pnp, rvec, tvec = cv2.solvePnP(objp, corners, self.cam_matrix, self.dist_coeffs)
            if not ret_pnp:
                logger.error("solvePnP计算失败！")
                return

            R_camera_board, _ = cv2.Rodrigues(rvec)
            M_camera_board = np.eye(4)
            M_camera_board[:3, :3] = R_camera_board
            M_camera_board[:3, 3] = tvec.flatten()

            print(f"成功计算出棋盘格位姿 M_camera_board:\n{M_camera_board}")

            objp_hom = np.hstack((objp, np.ones((objp.shape[0], 1))))
            points_in_camera_all = (M_camera_board @ objp_hom.T).T[:, :3]

            cam_file = os.path.join(self.save_dir, "all_points_camera_solvepnp.txt")
            np.savetxt(cam_file, points_in_camera_all, fmt="%.8f")
            print(f"所有 {len(points_in_camera_all)} 个角点的相机3D坐标 (solvePnP法) 已保存至: {cam_file}")

        finally:
            self.disconnect_camera()

    def collect_by_single_point_projection(self):
        """
        【方法二】: 通过手动点击像素点，利用深度图反投影计算单个点的3D坐标。
        此方法更依赖单点深度测量的精度。
        """
        print("\n" + "=" * 20 + " 开始使用【自动角点识别+反投影法】采集数据 " + "=" * 20)
        if not self.connect_camera():
            return

        print("请确保棋盘格在相机视野内清晰可见，且机器人已移开。")
        print("3秒后将自动捕获并处理图像...")
        time.sleep(3)

        try:
            # 1. 捕获对齐的帧
            frames = self.pipeline.wait_for_frames()
            align = rs.align(rs.stream.color)
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                logger.error("获取图像或深度帧失败！")
                raise Exception("获取图像或深度帧失败！")

            color_image = np.asanyarray(color_frame.get_data())
            gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            # 2. 自动查找棋盘格角点
            chessboard_size = self.config['chessboard_size']
            ret, corners = cv2.findChessboardCorners(gray_image, chessboard_size, None)
            if not ret:
                logger.error("在图像中未能找到棋盘格角点！请检查相机位置和光照。")
                raise

            # 亚像素级精化，得到更精确的 (u,v) 坐标
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray_image, corners, (11, 11), (-1, -1), criteria)

            print(f"成功检测到并精化了 {len(corners)} 个角点。")

            # 角度归一化
            corners = normalize_corner_order(corners, chessboard_size)
            # 显示角点
            cv2.drawChessboardCorners(color_image, chessboard_size, corners, ret)
            cv2.imshow("Detected Corners", color_image)
            cv2.waitKey(3000)
            cv2.destroyAllWindows()
            # 保存角点
            cv2.imwrite(os.path.join(self.save_dir, "projection_corners_detection.png"), color_image)

            # 3. 遍历每个角点，查询深度并反投影
            points_in_camera = []
            valid_corners_uv = []  # 存储那些深度有效的角点

            # corners 的形状是 (N, 1, 2)，我们需要 (N, 2)
            corners_flat = np.squeeze(corners)

            for i, (u, v) in enumerate(corners_flat):
                # 将浮点坐标四舍五入为整数，以便查询深度图
                u_int, v_int = int(round(u)), int(round(v))

                # 获取深度值
                depth = depth_frame.get_distance(u_int, v_int)

                if depth == 0:
                    logger.warning(f"角点 {i} (u,v)=({u:.2f}, {v:.2f}) 的深度值为0！该点将被忽略。")
                    continue

                # 使用精确的浮点 (u,v) 和深度值进行反投影
                point_camera = rs.rs2_deproject_pixel_to_point(self.intrinsics, [u, v], depth)
                points_in_camera.append(point_camera)
                valid_corners_uv.append((u, v))

                print(
                    f"[反投影法] 角点 {i}: (u,v)=({u:.2f}, {v:.2f}) -> 深度={depth:.4f}m -> P_cam={point_camera}")

            # 4. 保存结果
            points_cam_arr = np.array(points_in_camera)
            cam_file = os.path.join(self.save_dir, "all_points_camera_projection.txt")
            np.savetxt(cam_file, points_cam_arr, fmt="%.8f")
            print(f"采集到的 {len(points_cam_arr)} 个有效相机3D坐标 (自动反投影法) 已保存至: {cam_file}")
        finally:
            self.disconnect_camera()


def define_points_in_work_object_coordinate(pix_index=None, grid_size=0.018):
    """
    根据角点索引，定义它们在工件坐标系(Wobj)下的理想3D坐标。

    工件坐标系定义：
    - 原点 (0,0,0): 位于棋盘格的第一个内角点 (索引[0,0])。
    - X+ 轴: 沿着棋盘格的行方向 (水平向右)。
    - Y+ 轴: 沿着棋盘格的列方向 (垂直向下)。
    - Z+ 轴: 垂直于棋盘格平面向外 (根据右手定则)。

    参数:
    pix_index (np.ndarray): Nx2 的数组，包含目标角点的 (col, row) 索引。
    grid_size (float): 棋盘格每个格子的物理尺寸 (单位: 米)。

    返回:
    np.ndarray: Nx3 的数组，包含每个角点在工件坐标系下的 (x, y, z) 坐标。
    """
    print("--- 定义角点在工件坐标系(Wobj)下的坐标 ---")
    if pix_index is None:
        raise ValueError("pix_index 不能为空")
    if grid_size <= 0:
        raise ValueError("grid_size 必须大于0")

    # 我们期望的索引是 (col, row)，这直接对应于 (x, y)
    # pix_index 的第一列是 col (x方向)，第二列是 row (y方向)
    num_points = pix_index.shape[0]

    # 直接根据索引和格子尺寸计算x, y坐标
    # Z坐标在棋盘格平面上，因此始终为0
    points_in_wobj = np.zeros((num_points, 3))
    points_in_wobj[:, 0] = pix_index[:, 0] * grid_size  # X = col * size
    points_in_wobj[:, 1] = pix_index[:, 1] * grid_size  # Y = row * size

    # --- 验证步骤 ---
    print(f"选定的角点索引 (col, row):\n{pix_index}")
    print(f"计算出的工件坐标 (x, y, z) [米]:\n{points_in_wobj}")

    return points_in_wobj


def transform_points_form_work2base(points_in_wobj=None, M_base_wobj=None):
    """
    将工件坐标系下的点，通过变换矩阵，转换到基座坐标系下

    参数:
    points_in_wobj (np.ndarray): Nx3 的数组，来自上一步。
    M_base_wobj (np.ndarray): 4x4 的齐次变换矩阵，从工件坐标系(Wobj)到基座坐标系(Base)的变换，通过机械臂的3点标定确定后从示教器中读取
    注意:
    - 输入的变换矩阵 M_base_wobj 必须是 4x4 的齐次变换矩阵。
    - 输入的点云 points_in_wobj 必须是 Nx3 的数组，其中 N 是点的数量。
    返回:
    np.ndarray: Nx3 的数组，包含每个角点在基座坐标系下的 (x, y, z) 坐标。
    """
    if points_in_wobj is None or M_base_wobj is None:
        raise ValueError("points_in_wobj 和 M_base_wobj 不能为空")

    print("\n--- 将Wobj坐标变换到基座(Base)坐标 ---")

    # 1. 将 (N, 3) 的点云转换为 (N, 4) 的齐次坐标形式
    num_points = points_in_wobj.shape[0]
    points_in_wobj_hom = np.hstack((points_in_wobj, np.ones((num_points, 1))))

    # 2. 进行坐标变换
    # P_base = M_base_wobj * P_wobj
    # 为了高效计算，先计算变换矩阵的转置然后做点积
    points_in_base_hom = (M_base_wobj @ points_in_wobj_hom.T).T

    # 3. 将 (N, 4) 的齐次坐标转换回 (N, 3) 的笛卡尔坐标
    points_in_base = points_in_base_hom[:, :3]

    # --- 验证步骤 ---
    print(f"使用的变换矩阵 M_base_wobj:\n{M_base_wobj}")
    print(f"计算出的基座坐标 (x, y, z) [米]:\n{points_in_base}")

    return points_in_base


def calculate_tcp_by_sphere_fitting(list_flange_pose):
    """
    通过手动采集的多组法兰盘位姿，利用球面拟合计算TCP的 (x, y, z) 偏移。
    假设工具坐标系的姿态与法兰盘坐标系姿态相同 (Rx=Ry=Rz=0)。

    参数:
    list_flange_pose: 法兰盘位姿列表，每个元素是一个[x,y,z,rx,ry,rz]

    返回:
    np.ndarray: TCP的 (x, y, z) 偏移向量 (单位: 米)。
    """
    num_points = len(list_flange_pose)
    if num_points < 4:
        raise ValueError("至少需要4个不同姿态的位姿数据")

    list_M_base_flange = []
    for robot_pose in list_flange_pose:
        M_flange_board = robot_pose_to_homogeneous_matrix(robot_pose, order='ZYX')
        list_M_base_flange.append(M_flange_board)

    # 提取法兰盘的旋转矩阵 R 和平移向量 t
    rotations = [M[:3, :3] for M in list_M_base_flange]
    translations = [M[:3, 3] for M in list_M_base_flange]

    # 定义误差函数 (残差函数)
    def residuals(params):
        # params 包含了我们要求解的未知数：
        # 前3个是TCP在法兰盘坐标系下的偏移 [x, y, z]
        # 后3个是固定参考点在基座坐标系下的位置 [Px, Py, Pz]
        tcp_offset = params[0:3]
        tcp_offset[0] = 0
        tcp_offset[1] = 0
        reference_point_base = params[3:6]

        errors = []
        for i in range(num_points):
            R = rotations[i]
            t = translations[i]

            # 根据当前猜测的tcp_offset，计算出TCP在基座坐标系下的位置
            tcp_pos_base = t + R @ tcp_offset

            # 计算这个位置与猜测的固定参考点之间的距离误差
            error_vector = tcp_pos_base - reference_point_base
            errors.extend(error_vector)

        return np.array(errors)

    # 为求解器提供一个初始猜测值
    # 可以假设工具长度大约是200mm，并且参考点在某个大概的位置
    # 注意：这里的单位必须是米！
    initial_tcp_offset = np.array([0, 0, 0.045])
    # 用法兰盘位置的平均值作为参考点初始猜测
    initial_reference_point = np.mean(translations, axis=0)
    initial_params = np.concatenate([initial_tcp_offset, initial_reference_point])

    # 使用最小二乘法求解
    result = least_squares(residuals, initial_params)

    # 提取最终的TCP偏移结果
    optimal_tcp_offset = result.x[0:3]
    # 估计的参考点
    estimated_reference_point = result.x[3:6]
    print(f"最优的TCP偏移 (x, y, z): {optimal_tcp_offset}")
    print(f"估计的参考点 (Px, Py, Pz): {estimated_reference_point}")

    # --- 【新增】误差分析环节 ---
    print("\n" + "=" * 20 + " 拟合误差分析 " + "=" * 20)

    # 用计算出的最优解，来重新计算每个点的TCP位置
    calculated_tcp_positions = [t + R @ optimal_tcp_offset for R, t in zip(rotations, translations)]
    calculated_tcp_positions = np.array(calculated_tcp_positions)

    # 计算每个点到“最佳球心”（估计的参考点）的距离误差
    errors_m = np.linalg.norm(calculated_tcp_positions - estimated_reference_point, axis=1)
    errors_mm = errors_m * 1000.0  # 转换为毫米

    # 打印详细的每个点的残差
    print("--- 各姿态点的拟合残差 (单位: 毫米) ---")
    for i, error in enumerate(errors_mm):
        print(f"  姿态点 {i + 1}: 拟合误差 = {error:.4f} mm")

    # 计算并打包统计报告
    max_error = np.max(errors_mm)
    mean_error = np.mean(errors_mm)
    std_dev = np.std(errors_mm)

    error_report = {
        "max_residual_mm": max_error,
        "mean_residual_mm": mean_error,
        "std_dev_mm": std_dev
    }

    # 打印最终的统计报告
    print("\n--- 最终拟合精度统计 (单位: 毫米) ---")
    print(f"  最大残差 (Max Residual Error): {max_error:.4f} mm")
    print(f"  平均残差 (Mean Residual Error): {mean_error:.4f} mm")
    print(f"  残差标准差 (Standard Deviation): {std_dev:.4f} mm")

    return optimal_tcp_offset, error_report


def pixel_to_camera_coordinate(x, y, depth, K):
    """
    将像素坐标转换为相机坐标
    :param x: 像素坐标x
    :param y: 像素坐标y
    :param depth: 深度值
    :param K: 相机内参矩阵
    :return: 相机坐标
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # 计算相机坐标
    Zc = depth
    Xc = (x - cx) * Zc / fx
    Yc = (y - cy) * Zc / fy
    return np.array([Xc, Yc, Zc])


def camera_to_robot_coordinate(x_c, y_c, z_c, camera2world):
    """
        将相机坐标转换为机械臂基坐标
        :param x_c: 相机坐标x
        :param y_c: 相机坐标y
        :param z_c: 相机坐标z
        :param camera2world: 相机到机械臂基坐标系的变换矩阵 4*4
        :return: 机械臂基坐标
    """
    # 转换为齐次坐标
    camera_coord_homog = np.append([x_c, y_c, z_c], [1]).reshape(4, 1)
    # 转换到机器人基坐标系
    robot_coord = np.dot(camera2world, camera_coord_homog)
    robot_base_xyz = robot_coord[:3].flatten()  # 移除齐次坐标
    return robot_base_xyz


def do_calibrate_one_step():
    # --- 1. 配置参数 ---
    save_dir = os.path.join(BASE_DIR, "nine_point_calibrate_data")
    os.makedirs(save_dir, exist_ok=True)

    chessboard_size = (8, 8)
    config = {
        'camera_id': 246422072474,
        'save_dir': save_dir,
        'chessboard_size': chessboard_size,
        'chessboard_grid_size': 0.018,
        'num_points': 9
    }

    # 定义目标角点索引
    pix_index = np.array([
        [0, 0], [4, 0], [7, 0],
        [0, 3], [4, 3], [7, 3],
        [0, 7], [4, 7], [7, 7]
    ])

    points_in_base = [
        [288.0685, 54.43882, -31.32687],
        [288.0686, -6.361143, -30.49499],
        [290.4363, -51.35270, -30.49504],
        [227.2666, 51.17521, -23.19917],
        [227.2665, -6.104365, -20.25524],
        [230.7863, -49.36795, -21.27924],
        [151.2325, 46.18402, -10.71933],
        [150.3365, -6.743635, -8.543355],
        [153.9844, -46.48726, -9.119241],
    ]
    points_in_base = np.asarray(points_in_base) / 1000

    # # --- 2. 数据采集阶段 ---
    # # 运行其中一种方法，或者两种都运行以生成不同的数据文件
    # collector = CameraDataCollector(config)
    #
    # # === 运行方法一：SolvePnP ===
    # print(">>> 正在执行SolvePnP数据采集...")
    # collector.collect_by_solvepnp()
    #
    # # === 运行方法二：单点反投影 ===
    # print(">>> ---------------------------------")
    # print(">>> 正在执行单点反投影数据采集...")
    # collector.collect_by_single_point_projection()

    # --- 3. 计算阶段 ---
    # 输入已知的 M_base_wobj
    # robot_pose = [349.673, 47.6705, 151.804, -167.202, 1.04631, -88.6680]
    # robot_pose = [385.930, 35.9785, 167.968, -158.524, 1.68234, -90.0000]
    robot_pose = [333.333, 64.68000, 173.923, -157.037, 2.81215, -80.8984]
    M_base_wobj = robot_pose_to_homogeneous_matrix(robot_pose=robot_pose, order='ZYX')

    # 自动计算 P_base
    # # 步骤1: 定义Wobj坐标
    # points_in_wobj = define_points_in_work_object_coordinate(pix_index, config['chessboard_grid_size'])
    # # 步骤2: 变换到Base坐标
    # points_in_base = transform_points_form_work2base(points_in_wobj, M_base_wobj)
    # # 步骤3：基坐标补偿
    # xyz_scale = np.array([1.11496, 0.8479, 0.9253])
    # points_in_base[:, 0] = points_in_base[:, 0] * xyz_scale[0]
    # points_in_base[:, 1] = points_in_base[:, 1] * xyz_scale[1]
    # points_in_base[:, 2] = points_in_base[:, 2] * xyz_scale[2]

    # --- 4. 对比实验 ---
    # === 使用 SolvePnP 的数据进行计算 ===
    print("\n\n" + "#" * 20 + " 使用 SolvePnP 数据进行计算 " + "#" * 20)
    cam_file_solvepnp = os.path.join(save_dir, "all_points_camera_solvepnp.txt")
    if os.path.exists(cam_file_solvepnp):
        all_points_in_camera = np.loadtxt(cam_file_solvepnp)
        if len(all_points_in_camera) == chessboard_size[0] * chessboard_size[1]:
            # 获取列数
            num_cols = chessboard_size[0]
            linear_indices = pix_index[:, 1] * num_cols + pix_index[:, 0]
            points_in_camera = all_points_in_camera[linear_indices]
            # 计算刚性变换
            M_base_camera = rigid_transform(points_in_camera, points_in_base)
            print(f"\n[SolvePnP法] 计算出的 M_base_camera:\n{M_base_camera}")
            np.savetxt(os.path.join(save_dir, 'M_base_camera_by_solvepnp.txt'), M_base_camera, delimiter=' ',
                       fmt='%.8f')
            verify_transformation(M_base_camera, points_in_camera, points_in_base)
        else:
            logger.error(f"文件 {cam_file_solvepnp} 存在，但数据点数量与目标点数量不一致，无法进行计算。")
    else:
        logger.error(f"文件 {cam_file_solvepnp} 不存在，无法进行 SolvePnP 数据计算。")

    # === 使用单点反投影的数据进行计算 ===
    print("\n\n" + "#" * 20 + " 使用单点反投影数据进行计算 " + "#" * 20)
    cam_file_single = os.path.join(save_dir, "all_points_camera_projection.txt")
    if os.path.exists(cam_file_single):
        all_points_in_camera = np.loadtxt(cam_file_single)
        if len(all_points_in_camera) == chessboard_size[0] * chessboard_size[1]:
            # 获取列数
            num_cols = chessboard_size[0]
            linear_indices = pix_index[:, 1] * num_cols + pix_index[:, 0]
            points_in_camera = all_points_in_camera[linear_indices]
            # 计算刚性变换
            M_base_camera = rigid_transform(points_in_camera, points_in_base)
            np.savetxt(os.path.join(save_dir, 'M_base_camera_by_projection.txt'), M_base_camera, delimiter=' ',
                       fmt='%.8f')
            print(f"\n[单点法] 计算出的 M_base_camera:\n{M_base_camera}")
            verify_transformation(M_base_camera, points_in_camera, points_in_base)
        else:
            logger.error(f"文件 {cam_file_single} 存在，但数据点数量与目标点数量不一致，无法进行计算。")
    else:
        logger.error(f"文件 {cam_file_single} 不存在，无法进行单点反投影数据计算。")
    return save_dir


def do_calibrate_by_end_to_end(data_save_dir=None, redo=False):
    """
    通过RealSense相机验证手眼标定结果，支持用户点击图像获取3D坐标
    优化点：增加错误处理、可视化标记、深度平滑、结果保存和用户提示
    """
    # robot pose
    robot_pose_txt = os.path.join(data_save_dir, "robot_tcp_pose.txt")
    if not os.path.exists(robot_pose_txt):
        logger.error(f"文件 {robot_pose_txt} 不存在，无法进行手眼标定。")
        return
    camera_point_txt = os.path.join(data_save_dir, "end2end_camera_points.txt")

    robot_poses = np.loadtxt(robot_pose_txt, delimiter=' ')
    logger.info(f"读取到 {len(robot_poses)} 个机械臂位姿")

    # ========== 初始化相机 ==========
    print('正在连接相机...')
    camera = RealSenseCamera()
    camera.connect()
    print(f"相机连接成功!")
    print(f"相机内参:\n{camera.K}")

    # 存储像素坐标
    pixel_points = []
    # 存储相机坐标
    camera_points = []
    if os.path.exists(camera_point_txt):
        camera_points = np.loadtxt(camera_point_txt, delimiter=' ')
        logger.info(f"读取到 {len(camera_points)} 个相机坐标")
    else:
        logger.warning(f"文件 {camera_point_txt} 不存在，将开始手动获取相机坐标。")
        camera_points = []

    if redo or len(camera_points) == 0:
        # ========== 鼠标回调函数 ==========
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
                pixel_points.append([x, y])
                # 1.从对齐的深度图获取深度值（注意坐标顺序）
                depth_value = depth[y, x]
                depth_value = depth_value[0]
                if depth_value < 0.1 or depth_value > 0.75:  # 合理深度范围判断
                    print(f"深度值({depth_value:.3f}m)超出有效范围(0.1-0.71m)")
                    return
                # 2. 将像素点投影到相机坐标系
                camera_xyz = pixel_to_camera_coordinate(x, y, depth_value, camera.K)
                camera_xyz = camera_xyz.flatten()
                camera_points.append(camera_xyz)
                print(
                    f"点击位置: ({x},{y}) → 深度: {depth_value:.3f}m → 相机坐标: X={camera_xyz[0]:.4f}m, Y={camera_xyz[1]:.4f}m, Z={camera_xyz[2]:.4f}m")

        # ========== 主循环==========
        try:
            print("\n操作说明:")
            print("1. 点击图像上的点获取其在相机坐标系中的坐标值")
            print("2. 按 'r' 清除所有测量结果")
            print("3. 按 'ESC' 退出程序")
            while True:
                # 获取图像
                images = camera.get_image_bundle()
                if not images or 'rgb' not in images or 'aligned_depth' not in images:
                    print("获取图像失败，重试...")
                    continue
                # 转换色彩空间以适应OpenCV显示
                rgb = cv2.cvtColor(images['rgb'], cv2.COLOR_RGB2BGR)
                depth = images['aligned_depth']
                # 显示操作提示
                cv2.putText(rgb, "ESC:exit | r:clear | h:home", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                # 显示窗口并绑定鼠标事件
                cv2.imshow('VerifyCalibration', rgb)
                cv2.setMouseCallback('VerifyCalibration', on_mouse, param={'rgb': rgb, 'aligned_depth': depth})
                # 处理键盘事件
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC 退出
                    print("程序退出")
                    break
                elif key == ord('s'):  # 保存结果
                    pixel_points.clear()
                    camera_points.clear()
                    print("已清除所有测量结果")
        except Exception as e:
            print(f"程序运行出错: {str(e)}")
        finally:
            # 资源清理
            cv2.destroyAllWindows()
            camera.disconnect()

    if len(robot_poses) != len(camera_points):
        logger.error(f"机械臂位姿数量({len(robot_poses)})与相机坐标数量({len(camera_points)})不一致，无法进行手眼标定。")
        return

    # ========== 计算手眼标定矩阵 ==========
    # robot_poses 每个元素都只提取前3个元素
    robot_poses = robot_poses[:, :3] / 1000
    # 计算刚性变换
    camera_points = np.asarray(camera_points)
    M_base_camera = rigid_transform(camera_points, robot_poses)
    print(f"\n[end2end法] camera_points:\n{camera_points}")
    print(f"\n[end2end法] 计算出的 M_base_camera:\n{M_base_camera}")
    np.savetxt(os.path.join(save_dir, 'M_base_camera_by_end2end.txt'), M_base_camera, delimiter=' ', fmt='%.8f')
    np.savetxt(os.path.join(save_dir, 'end2end_camera_points.txt'), camera_points, delimiter=' ', fmt='%.8f')
    verify_transformation(M_base_camera, camera_points, robot_poses)


def verify_calibration_by_realsense_camera(data_save_dir=None, move_robot=False):
    """
    通过RealSense相机验证手眼标定结果，支持用户点击图像获取3D坐标
    增加错误处理、可视化标记、深度平滑、结果保存和用户提示
    """
    if not data_save_dir:
        logger.error(f"数据保存文件夹未指定")
        return
    # ========== 读取标定结果 ==========
    txt_name = 'M_base_camera_by_end2end.txt'
    # txt_name = 'M_base_camera_by_projection.txt'
    M_base_camera = np.loadtxt(os.path.join(data_save_dir, txt_name), delimiter=' ')
    print(f"手眼标定矩阵:\n{M_base_camera}")

    # 存储测量结果
    measurement_results = []

    # ========== 初始化相机 ==========
    print('正在连接相机...')
    camera = RealSenseCamera()
    camera.connect()
    print(f"相机连接成功!")
    print(f"相机内参:\n{camera.K}")

    # ========== 初始化夹爪 ==========
    if move_robot:
        print('正在连接夹爪...')
        gripper = GripperWiFiController()
        if not gripper.connect():
            print(f"夹爪连接失败")
            return
        print(f"夹爪连接成功!")

    # ========== 初始化机械臂 ==========
    robot = DensorRobot()
    # default_grasp_pose = [140, 0, 230.0, -167, 2, 81, 5]
    # home position
    default_grasp_pose = [130, 0, 333.0, -171, -16, 171, 5]
    # 松开夹爪的位姿
    open_grasp_pose = [165, -265, 210.0, -171, -16, 171, 5]
    # y+ pose
    y_plus_pose = [-171, -16, 171, 5]
    # y- pose
    y_minus_pose = [-179, -16, 171, 5]
    if move_robot:
        print('正在连接机械臂...')
        if not robot.connect():
            logger.error("机械臂连接失败")
            return
        print(f"机械臂连接成功! 移动到安全点")
        robot.send_position(default_grasp_pose)
        time.sleep(1)

    # ========== 鼠标回调函数 ==========
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

            # 1.从对齐的深度图获取深度值（注意坐标顺序）
            depth_value = depth[y, x]
            depth_value = depth_value[0]
            if depth_value < 0.1 or depth_value > 0.75:  # 合理深度范围判断
                print(f"深度值({depth_value:.3f}m)超出有效范围(0.1-0.71m)")
                return
            # 2. 将像素点投影到相机坐标系
            camera_xyz = pixel_to_camera_coordinate(x, y, depth_value, camera.K)
            # 3. 将相机坐标转换为机械臂基坐标
            robot_base_xyz = camera_to_robot_coordinate(camera_xyz[0], camera_xyz[1],
                                                        camera_xyz[2], M_base_camera)
            # m -> mm
            robot_base_xyz = robot_base_xyz * 1000
            # 5. 缩放
            center_points = np.array([220, 0, 0])
            # xyz_scale = np.array([1.11496, 0.8479, 0.9253])
            scale_robot_base_xyz = center_points + (robot_base_xyz - center_points) * np.array([1.0, 0.8, 1.15])
            # 4. 显示和记录结果
            result_str = (f"像素点: ({x},{y}) → 深度: {depth_value:.3f}m → "
                          f"相机坐标: X={camera_xyz[0]:.4f}m, Y={camera_xyz[1]:.4f}m, Z={camera_xyz[2]:.4f}m → "
                          f"基座坐标: X={robot_base_xyz[0]:.4f}mm, Y={robot_base_xyz[1]:.4f}mm, Z={robot_base_xyz[2]:.4f}mm → "
                          f"缩放坐标: X={scale_robot_base_xyz[0]:.4f}mm, Y={scale_robot_base_xyz[1]:.4f}mm, Z={scale_robot_base_xyz[2]:.4f}mm")
            print(result_str)
    
            # scale_robot_base_xyz 四舍五入，保留两位小数点
            scale_robot_base_xyz = np.round(scale_robot_base_xyz, 2)
            # scale_robot_base_xyz = np.round(robot_base_xyz, 2)
            # 6. 移动机械臂到点击点
            if move_robot:
                # # z轴限制
                # if scale_robot_base_xyz[2] < -28:
                #     scale_robot_base_xyz[2] = scale_robot_base_xyz[2] - 1
                # if scale_robot_base_xyz[2] < -26:
                #     scale_robot_base_xyz[2] = scale_robot_base_xyz[2] - 2
                # elif scale_robot_base_xyz[2] < -25:
                #     scale_robot_base_xyz[2] = scale_robot_base_xyz[2] - 3
                # elif scale_robot_base_xyz[2] < -24:
                #     scale_robot_base_xyz[2] = scale_robot_base_xyz[2] - 2
                # elif scale_robot_base_xyz[2] < -20:
                #     scale_robot_base_xyz[2] = scale_robot_base_xyz[2] - 5
                # elif scale_robot_base_xyz[2] < 0:
                #     scale_robot_base_xyz[2] = scale_robot_base_xyz[2] - 10
                # else:
                #     scale_robot_base_xyz[2] = scale_robot_base_xyz[2] - 2
                # scale_robot_base_xyz[2] = max(scale_robot_base_xyz[2], -30)
                # scale_robot_base_xyz[2] = map_x_to_z(scale_robot_base_xyz[0])
                # if scale_robot_base_xyz[1] < 0:
                #     scale_robot_base_xyz[1] = scale_robot_base_xyz[1] + 10\
                scale_robot_base_xyz[2] = scale_robot_base_xyz[2] - 5
                print(f"优化后的坐标点: {scale_robot_base_xyz}")
                z_up_diff = 50
                # 先回到安全点
                robot.send_position(default_grasp_pose)
                time.sleep(2)
                # 移动到点击点
                if scale_robot_base_xyz[1] < 0:
                    robot_pose = list(scale_robot_base_xyz) + y_minus_pose
                else:
                    robot_pose = list(scale_robot_base_xyz) + y_plus_pose
                print(f"目标点位姿: {robot_pose}")
                # 停留在上方
                robot_pose[2] = robot_pose[2] + z_up_diff
                print(f"移动到目标上方: {robot_pose}")
                robot.send_position(robot_pose)
                # 打开夹爪
                print("打开夹爪...")
                gripper.open()
                time.sleep(2)

                # # 旋转角度30度
                # robot.rotate_relative_angle(30, j_num=6)
                # time.sleep(1)
                # cur_pose = robot.get_current_position()
                # print(f"当前机械臂位置: {cur_pose}")

                # 移动到点击点
                robot_pose[2] = robot_pose[2] - z_up_diff
                print(f"移动到目标点位姿: {robot_pose}")
                robot.send_position(robot_pose)
                time.sleep(2)
                # 关闭夹爪
                print("关闭夹爪...")
                gripper.close()
                time.sleep(1)
                # 回到安全点
                print("回到安全点...")
                robot.send_position(default_grasp_pose)
                time.sleep(2)
                # 松开夹爪点
                print("前往夹爪松开点")
                robot.send_position(open_grasp_pose)
                time.sleep(2)
                # 松开夹爪
                print("松开夹爪...")
                gripper.open()
                time.sleep(1)
                # 回到安全点
                print("回到安全点...")
                robot.send_position(default_grasp_pose)
                time.sleep(2)

            # 记录结果
            measurement_results.append({
                "pixel": (x, y),
                "depth": depth_value,
                "camera_coords": (camera_xyz[0], camera_xyz[1], camera_xyz[2]),
                "base_coords": (robot_base_xyz[0], robot_base_xyz[1], robot_base_xyz[2]),
                "depth_origin": depth_value,
            })

            # 5. 在图像上标记点击点和信息
            cv2.circle(rgb, (x, y), 5, (0, 0, 255), -1)  # 红色圆点标记
            cv2.putText(rgb, f"X:{robot_base_xyz[0]:.3f}", (x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(rgb, f"Y:{robot_base_xyz[1]:.3f}", (x + 10, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(rgb, f"Z:{robot_base_xyz[2]:.3f}", (x + 10, y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # ========== 主循环==========
    try:
        print("\n操作说明:")
        print("1. 点击图像上的点获取其在机械臂基座坐标系中的坐标")
        print("3. 按 'r' 清除所有测量结果")
        print("4. 按 'ESC' 退出程序")

        while True:
            # 获取图像
            images = camera.get_image_bundle()
            if not images or 'rgb' not in images or 'aligned_depth' not in images:
                print("获取图像失败，重试...")
                continue

            # 转换色彩空间以适应OpenCV显示
            rgb = cv2.cvtColor(images['rgb'], cv2.COLOR_RGB2BGR)
            depth = images['aligned_depth']

            # 显示操作提示
            cv2.putText(rgb, "ESC:exit | s:save | r:clear | h:home", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # 显示窗口并绑定鼠标事件
            cv2.imshow('VerifyCalibration', rgb)
            cv2.setMouseCallback('VerifyCalibration',
                                 on_mouse, param={'rgb': rgb, 'aligned_depth': depth})

            # 处理键盘事件
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC 退出
                print("程序退出")
                break
            elif key == ord('r'):  # 清除结果
                measurement_results.clear()
                print("已清除所有测量结果")
            elif key == ord('h') or key == ord('H'):  # 返回抓取默认点
                # 执行返回默认抓取点的逻辑
                if move_robot:
                    robot.send_position(default_grasp_pose)
                    time.sleep(2)
                    print("已返回抓取默认点")

    except Exception as e:
        print(f"程序运行出错: {str(e)}")
    finally:
        # 资源清理
        cv2.destroyAllWindows()
        camera.disconnect()
        if move_robot:
            robot.send_position(default_grasp_pose)
            time.sleep(0.2)
            robot.close()


def test_calculate_tcp_by_sphere_fitting():
    list_flange_pose = [
        # [352.7166, 3.834210, 275.4843, 177.1057, -7.580472, -151.7946],
        # [381.6855, -0.4848859, 269.3588, -164.2730, 36.22287, -44.85877],
        # [341.0879, 50.23175, 244.1174, 112.4219, -10.66462, 27.04406],
        # [360.0086, 28.10650, 275.0159, -179.6946, 0.9222620, -4.622844],
        # [313.5419, 4.693391, 252.9166, -108.9253, -62.30903, -28.31354],
        # [335.3222, -10.97202, 249.9016, -72.89448, -64.41283, -31.75434],
        # [306.2386, 4.831776, 240.7068, -87.02254, 11.26474, -84.24688],
        # [309.6485, 7.596585, 230.6693, -67.90858, 15.67651, -86.06684],
        # [310.6455, 19.23397, 257.2483, 156.6221, 7.133883, 111.4414],
        # [312.3597, 20.28491, 222.0596, 62.31935, -42.56069, 132.3357]

        # [300.6935, 5.897338, 280.2787, -173.5956, 2.963190, -113.6239],
        # [305.6845, 30.23043, 273.2603, -152.3229, -38.03855, -120.5387],
        # [319.1454, -26.62746, 266.1335, 151.3848, 60.70327, -118.5182],
        # [357.2262, 6.396538, 283.5108, 137.8029, -9.319727, -99.86958],
        # [277.5786, -6.563340, 228.6363, 64.37449, -33.45845, 130.8999],
        # [292.6084, 28.54366, 226.4671, 17.30167, 59.50727, -23.84530],
        # [305.8756, 3.380453, 282.0353, 176.9817, -4.492258, 102.2633],
        # [327.4564, 18.15222, 284.6125, -164.1305, 20.73775, 87.55800]

        [289.9453, -10.17129, 195.7473, 179.1251, -0.8051122, -100.7155],
        [249.2397, -9.187646, 166.5136, 164.7294, -71.82033, 15.27701],
        [258.0807, -14.32081, 150.7769, 80.46963, 30.34707, 95.07211],
        [289.0140, -36.57229, 142.0376, 64.18985, 3.800748, 143.4792],
        [277.0410, -29.79131, 182.3373, -129.9759, 15.44075, -40.07043],
        [312.9283, -41.63756, 179.0982, -119.5534, 28.63601, -3.005495],
        [328.0746, -48.26571, 160.1607, -87.94541, 32.68209, 17.23763],
        [264.3297, -25.56550, 156.0101, -85.87966, 40.58895, -53.78360],
        [286.5349, -16.73933, 194.4251, -170.4328, -8.955682, 27.43024],
        [297.8774, 18.40981, 177.9461, 117.5849, 24.27140, 15.89549],
        [333.6035, 19.61710, 173.5613, -110.0190, 36.61344, 137.6264],
        [352.9947, -8.362139, 185.0550, -123.8039, 9.795426, 89.75771]
    ]
    calculate_tcp_by_sphere_fitting(list_flange_pose)


# ==============================================================================
#                                  主函数
# ==============================================================================
if __name__ == '__main__':
    save_dir = os.path.join(BASE_DIR, "nine_point_calibrate_data")
    # 标定
    # do_calibrate_one_step()
    # 端到端标定
    # do_calibrate_by_end_to_end(save_dir)
    # 验证
    verify_calibration_by_realsense_camera(data_save_dir=save_dir, move_robot=True)
