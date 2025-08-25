import logging
import os
import time

import colorlog
import cv2
import numpy as np
import pyrealsense2 as rs

from calibrate.utils import normalize_corner_order, robot_pose_to_homogeneous_matrix


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


# TCP 四点标定
# 原则一：姿态差异最大化 (最重要)
# 错误做法: 从四个非常接近的角度去触碰参考点。这就像只从正面给一个人拍照，你无法准确判断他的鼻子有多长。
# 正确做法: 从四个差异巨大、接近相互垂直的方向去触碰同一个参考点。
# 点1: 从参考点的正上方垂直向下触碰。
# 点2: 从参考点的正前方水平向后触碰。
# 点3: 从参考点的左侧水平向右触碰。
# 点4: 从一个斜下方45度的角度向上触碰。
# 原理: 姿态差异越大，构成的球面约束就越强，算法解算出的球心（即TCP）就越精确。如果能采集更多点（例如六点法），可以从更多角度去逼近。
# 原则二：触点精确
# 在示教器上使用最低速、最精细的移动模式（通常是0.1mm步进）。
# 从多个视角反复观察，确保是探针的尖端正中心触碰到了参考点的正中心。不要用探针的侧面去触碰。
# 触碰到位后，确保机械臂完全静止，再在示教器上记录该点。
# 原则三：保持工具不变
# 在整个TCP标定过程中，绝对不要重新安装或紧固您的探针。

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
    M[:3, 3] = t
    return M


def verify_transformation(M, points_A, points_B):
    errors = []
    for i in range(points_A.shape[0]):
        p_A_hom = np.append(points_A[i], 1)
        p_B_predicted = (M @ p_A_hom)[:3]
        p_B_actual = points_B[i]
        error = np.linalg.norm(p_B_predicted - p_B_actual)
        errors.append(error)
        logger.info(
            f"点 {i + 1}: 实际 P_base = {p_B_actual}, 预测 P_base = {p_B_predicted}, 误差 = {error * 1000:.4f} mm")
    errors = np.array(errors)
    logger.info("\n" + "=" * 20 + " 验证结果总结 " + "=" * 20)
    logger.info(f"平均重投影误差: {np.mean(errors) * 1000:.4f} mm")
    logger.info(f"最大重投影误差: {np.max(errors) * 1000:.4f} mm")
    logger.info(f"误差标准差: {np.std(errors) * 1000:.4f} mm")
    logger.info(f"RMSE 误差: {np.sqrt(np.mean(errors ** 2)) * 1000:.4f} mm")


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
            config.enable_device(self.config['camera_id'])
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

            logger.info("Realsense相机连接成功。")
            logger.info(f"使用相机内参: \n{self.cam_matrix}")
            logger.info(f"使用相机畸变系数: {self.dist_coeffs}")
            return True
        except Exception as e:
            logger.error(f"连接Realsense相机失败: {e}")
            return False

    def disconnect_camera(self):
        """断开相机连接"""
        if self.pipeline:
            self.pipeline.stop()
            self.pipeline = None
            logger.info("相机已关闭。")

    def collect_by_solvepnp(self):
        """
        【方法一】: 使用solvePnP整体解算棋盘格位姿，计算所有角点的3D坐标。
        这是理论上更精确、更稳健的方法。
        """
        logger.info("\n" + "=" * 20 + " 开始使用 solvePnP 方法采集数据 " + "=" * 20)
        if not self.connect_camera():
            return

        logger.info("请确保棋盘格在相机视野内清晰可见，且机器人已移开。")
        logger.info("5秒后将自动捕获图像...")
        time.sleep(5)

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

            logger.info(f"成功计算出棋盘格位姿 M_camera_board:\n{M_camera_board}")

            objp_hom = np.hstack((objp, np.ones((objp.shape[0], 1))))
            points_in_camera_all = (M_camera_board @ objp_hom.T).T[:, :3]

            cam_file = os.path.join(self.save_dir, "all_points_camera_solvepnp.txt")
            np.savetxt(cam_file, points_in_camera_all, fmt="%.8f")
            logger.info(f"所有 {len(points_in_camera_all)} 个角点的相机3D坐标 (solvePnP法) 已保存至: {cam_file}")

        finally:
            self.disconnect_camera()

    def collect_by_single_point_projection(self):
        """
        【方法二】: 通过手动点击像素点，利用深度图反投影计算单个点的3D坐标。
        此方法更依赖单点深度测量的精度。
        """
        logger.info("\n" + "=" * 20 + " 开始使用【自动角点识别+反投影法】采集数据 " + "=" * 20)
        if not self.connect_camera():
            return

        logger.info("请确保棋盘格在相机视野内清晰可见，且机器人已移开。")
        logger.info("5秒后将自动捕获并处理图像...")
        time.sleep(5)

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

            logger.info(f"成功检测到并精化了 {len(corners)} 个角点。")

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

                logger.info(
                    f"[反投影法] 角点 {i}: (u,v)=({u:.2f}, {v:.2f}) -> 深度={depth:.4f}m -> P_cam={point_camera}")

            # 4. 保存结果
            points_cam_arr = np.array(points_in_camera)
            cam_file = os.path.join(self.save_dir, "all_points_camera_projection.txt")
            np.savetxt(cam_file, points_cam_arr, fmt="%.8f")
            logger.info(f"采集到的 {len(points_cam_arr)} 个有效相机3D坐标 (自动反投影法) 已保存至: {cam_file}")
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
    logger.info("--- 定义角点在工件坐标系(Wobj)下的坐标 ---")
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
    logger.info(f"选定的角点索引 (col, row):\n{pix_index}")
    logger.info(f"计算出的工件坐标 (x, y, z) [米]:\n{points_in_wobj}")

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

    logger.info("\n--- 将Wobj坐标变换到基座(Base)坐标 ---")

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
    logger.info(f"使用的变换矩阵 M_base_wobj:\n{M_base_wobj}")
    logger.info(f"计算出的基座坐标 (x, y, z) [米]:\n{points_in_base}")

    return points_in_base


# ==============================================================================
#                                  主函数
# ==============================================================================
if __name__ == '__main__':

    # --- 1. 配置参数 ---
    save_dir = os.path.join(BASE_DIR, "nine_point_calibrate_data")
    os.makedirs(save_dir, exist_ok=True)

    chessboard_size = (8, 8)
    config = {
        'camera_id': 246422072474,
        'save_dir': save_dir,
        'chessboard_size': chessboard_size,
        'chessboard_grid_size': 0.018,
        'num_points': 9  # 仅用于单点法
    }

    # 定义目标角点索引
    pix_index = np.array([
        [0, 0], [3, 0], [6, 0],
        [0, 3], [3, 3], [6, 3],
        [0, 6], [3, 6], [6, 6]
    ])

    # --- 2. 数据采集阶段 ---
    # 运行其中一种方法，或者两种都运行以生成不同的数据文件

    collector = CameraDataCollector(config)

    # === 运行方法一：SolvePnP ===
    logger.info(">>> 正在执行SolvePnP数据采集...")
    collector.collect_by_solvepnp()

    # === 运行方法二：单点反投影 ===
    logger.info(">>> ---------------------------------")
    logger.info(">>> 正在执行单点反投影数据采集...")
    collector.collect_by_single_point_projection()

    # --- 3. 计算阶段 ---
    # 输入已知的 M_base_wobj
    M_base_wobj = np.array([
        [1, 0, 0, 0.5],
        [0, 1, 0, 0.1],
        [0, 0, 1, 0.2],
        [0, 0, 0, 1]
    ])
    # M_base_wobj = robot_pose_to_homogeneous_matrix(robot_pose=[x, y, z, rx, ry, rz], order='ZYX')

    # 自动计算 P_base
    # 步骤1: 定义Wobj坐标
    points_in_wobj = define_points_in_work_object_coordinate(pix_index, config['chessboard_grid_size'])
    # 步骤2: 变换到Base坐标
    points_in_base = transform_points_form_work2base(points_in_wobj, M_base_wobj)

    # --- 4. 对比实验 ---
    # === 使用 SolvePnP 的数据进行计算 ===
    logger.info("\n\n" + "#" * 20 + " 使用 SolvePnP 数据进行计算 " + "#" * 20)
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
            logger.info(f"\n[SolvePnP法] 计算出的 M_base_camera:\n{M_base_camera}")
            verify_transformation(M_base_camera, points_in_camera, points_in_base)
        else:
            logger.error(f"文件 {cam_file_solvepnp} 存在，但数据点数量与目标点数量不一致，无法进行计算。")
    else:
        logger.error(f"文件 {cam_file_solvepnp} 不存在，无法进行 SolvePnP 数据计算。")

    # === 使用单点反投影的数据进行计算 ===
    logger.info("\n\n" + "#" * 20 + " 使用单点反投影数据进行计算 " + "#" * 20)
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
            logger.info(f"\n[单点法] 计算出的 M_base_camera:\n{M_base_camera}")
            verify_transformation(M_base_camera, points_in_camera, points_in_base)
        else:
            logger.error(f"文件 {cam_file_single} 存在，但数据点数量与目标点数量不一致，无法进行计算。")
    else:
        logger.error(f"文件 {cam_file_single} 不存在，无法进行单点反投影数据计算。")
