# coding=utf-8
"""
眼在手外 用采集到的图片信息和机械臂位姿信息计算相机坐标系相对于机械臂基座标的旋转矩阵和平移向量
"""

import os.path

import cv2
import numpy as np

from hardware.camera import RealSenseCamera

np.set_printoptions(precision=8, suppress=True)


def euler_angles_to_rotation_matrix(rx, ry, rz):
    # 计算旋转矩阵
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
    R = Rz @ Ry @ Rx
    # R = Rx @ Ry @ Rz
    return R


def pose_to_homogeneous_matrix(pose):
    x, y, z, rx, ry, rz = pose
    R = euler_angles_to_rotation_matrix(rx, ry, rz)
    t = np.array([x, y, z]).reshape(3, 1)
    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = t[:, 0]
    return H


def inverse_transformation_matrix(T):
    """
    求 4x4 齐次变换矩阵的逆
    """
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError("输入必须是 4x4 齐次矩阵")

    R = T[:3, :3]
    t = T[:3, 3]

    # 检查R是否近似正交
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-8):
        raise ValueError("旋转部分不是正交矩阵，输入可能不是合法SE(3)变换")

    R_inv = R.T
    t_inv = -R_inv @ t

    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv


def compute_reprojection_error(obj_points, img_points, rvecs, tvecs, K, dist):
    """
    计算重投影误差（像素）
    - obj_points: [M][Ni,3]  每张图棋盘3D角点
    - img_points: [M][Ni,1,2] 或 [M][Ni,2]  棋盘检测到的2D角点
    - rvecs/tvecs: [M]  对应每张图的PnP结果（或 calibrateCamera 的输出）
    - K, dist: 相机内参与畸变
    """
    assert len(obj_points) == len(img_points) == len(rvecs) == len(tvecs)

    total_sq_err = 0.0
    total_pts = 0
    per_view_rms = []

    for i in range(len(obj_points)):
        # 投影 3D -> 2D
        proj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist)  # (Ni,1,2) float64
        proj = proj.reshape(-1, 2).astype(np.float64)

        # 观测角点统一为 (Ni,2) float64
        obs = img_points[i]
        if obs.ndim == 3 and obs.shape[1] == 1:
            obs = obs.reshape(-1, 2)
        obs = obs.astype(np.float64)

        # 残差 (Ni,2)
        residual = obs - proj
        # 每张图 RMS：sqrt(mean(||e||^2))
        rms_i = np.sqrt(np.mean(np.sum(residual ** 2, axis=1)))
        per_view_rms.append(rms_i)

        total_sq_err += np.sum(residual ** 2)
        total_pts += residual.shape[0]

    global_rms = np.sqrt(total_sq_err / total_pts)
    return global_rms, per_view_rms


# 计算相机坐标系相到机械臂基座标的旋转矩阵和平移向量
def compute_T(images_dir, pose_txt, corner_point_long, corner_point_short, corner_point_size):
    print("手眼标定采集的标定版图片所在路径", images_dir)
    print("手眼标定采集的机械臂位姿信息所在路径", pose_txt)
    print("标定板的中长度对应的角点的个数", corner_point_long)
    print("标定板的中宽度对应的角点的个数", corner_point_short)
    print("标定板一格的长度/m", corner_point_size)

    if not os.path.isdir(images_dir):
        print("标定板图片所在路径不存在")
        return
    if not os.path.isfile(pose_txt):
        print("机械臂位姿信息文件不存在")
        return

    # 1.标定板图片排序
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    checkerboard_image_files = [f for f in os.listdir(images_dir) if os.path.splitext(f.lower())[1] in exts]
    if not checkerboard_image_files:
        raise RuntimeError(f"在 {images_dir} 未找到图片！")

    def _num_key(name):
        # 尽量按数值前缀排序（如 12_x.png -> 12），否则用文件名
        stem = os.path.splitext(name)[0]
        try:
            return int(''.join([c for c in stem if c.isdigit()]))
        except:
            return stem

    checkerboard_image_files.sort(key=_num_key)

    # 2.读取机械臂TCP位姿，构造gripper2base的4x4齐次转换矩阵 (顺序要和图片保持一致)
    M_gripper2base_matrices = []
    with open(pose_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()  # 兼容多空白/Tab
            if len(parts) < 6:
                raise ValueError(f"位姿行字段不足6个: {line}")
            x, y, z, rx, ry, rz = map(float, parts[:6])  # 单位约定：米 & 弧度
            # 获取 gripper2base 变换矩阵
            M_gripper2base_matrices.append(pose_to_homogeneous_matrix([x, y, z, rx, ry, rz]))

    if len(M_gripper2base_matrices) != len(checkerboard_image_files):
        print(f"警告：TCP位姿数量与图片数量不一致，请确认两者顺序与数量一致。")
        return

    # 3.设置标定板坐标系下的点坐标，所有点的Z坐标全部为0，所以只需要赋值x和y
    objp = np.zeros((corner_point_long * corner_point_short, 3), np.float32)
    objp[:, :2] = np.mgrid[0:corner_point_long, 0:corner_point_short].T.reshape(-1, 2) * corner_point_size

    obj_points = []  # 存储3D点
    img_points = []  # 存储2D点

    # 4.查找图片的角点
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
    for filename in checkerboard_image_files:
        checkerboard_image = os.path.join(images_dir, filename)
        if os.path.exists(checkerboard_image):
            img = cv2.imread(checkerboard_image)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            size = gray.shape[::-1]
            ret, corners = cv2.findChessboardCorners(gray, (corner_point_long, corner_point_short), None)
            if ret:
                # 添加标定板坐标系下的3D空间点
                obj_points.append(objp)
                # 优化角点坐标，在原角点的基础上寻找亚像素角点
                corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
                # 检查角点方向
                if abs(corners[0][0][1] - corners[1][0][1]) > 10:
                    print(f"图片{filename} 角点识别方向与标定板坐标系方向不一致，请处理")
                    return
                # 添加像素坐标系下的2D空间点
                img_points.append(corners)

    # 5.相机标定，获取相机内参矩阵、畸变系数，每组图的标定板坐标系到相机坐标系的旋转平移矩阵
    print(f"开始进行相机标定 图片个数：{len(img_points)}")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    print("相机内参矩阵:\n", mtx)  # 内参数矩阵
    print("畸变系数:\n", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
    # 保存内参矩阵到./calibrate_result/mtx.txt
    np.savetxt("./calibrate_result/camera_matrix.txt", mtx)
    # 保存畸变系数到./calibrate_result/dist.txt
    np.savetxt("./calibrate_result/distortion_coefficients.txt", dist)

    # # 读取内参矩阵
    # mtx = np.loadtxt("./calibrate_result/camera_matrix.txt")
    # dist = np.loadtxt("./calibrate_result/distortion_coefficients.txt")
    # rvecs = []
    # tvecs = []
    # for i in range(len(img_points)):
    #     ret, rvec, tvec = cv2.solvePnP(obj_points[i], img_points[i], mtx, dist)
    #     rvecs.append(rvec)
    #     tvecs.append(tvec)

    # 6. 获取机械臂臂基坐标系到机器人末端的变换矩阵，对M_gripper2base求逆
    R_base2gripper = []
    t_base2gripper = []
    for M_gripper2base in M_gripper2base_matrices:
        M_base2gripper = inverse_transformation_matrix(M_gripper2base)
        # M_base2gripper = M_gripper2base
        R_base2gripper.append(M_base2gripper[0:3, 0:3])
        t_base2gripper.append(M_base2gripper[0:3, 3])

    # 7. 调用 cv2.calibrateHandEye 进行手眼标定 (方法: TSAI)
    print("开始进行手眼标定....")
    methods_dict = {
        cv2.CALIB_HAND_EYE_TSAI: "TSAI",
        cv2.CALIB_HAND_EYE_PARK: "PARK",
        cv2.CALIB_HAND_EYE_HORAUD: "HORAUD",
        cv2.CALIB_HAND_EYE_ANDREFF: "ANDREFF",
        cv2.CALIB_HAND_EYE_DANIILIDIS: "DANIILIDIS",
    }
    R_cam2base = None
    t_cam2base = None
    for method in methods_dict:
        print(f"\n\n开始进行手眼标定 方法：{methods_dict[method]}")
        R_cam2base, t_cam2base = cv2.calibrateHandEye(
            R_base2gripper,
            t_base2gripper,
            rvecs,
            tvecs,
            method=method,
        )
        print(f"[{methods_dict[method]}] 相机坐标系到机械臂基坐标系的旋转矩阵:")
        print(R_cam2base)
        print("[{methods_dict[method]}] 相机坐标系到机械臂基坐标系的平移向量:")
        print(t_cam2base)

    # 选择最优结果 可自行选择
    return R_cam2base, t_cam2base, mtx, dist


# 验证标定结果by图片
def verify_calibration_by_image(image_path, R_cam2base, T_cam2base,
                                corner_point_long, corner_point_short, square_size,
                                camera_matrix, dist_coeffs):
    """
    验证标定结果
    :param image_path: 验证用的棋盘格图像
    :param R_cam2base: 相机到基座的旋转矩阵 (3x3)
    :param T_cam2base: 相机到基座的平移向量 (3x1)
    :param corner_point_long: 棋盘格长边角点数
    :param corner_point_short: 棋盘格短边角点数
    :param square_size: 棋盘格格子物理尺寸 (m)
    :param camera_matrix: 相机内参矩阵
    :param dist_coeffs: 畸变系数
    """
    # 加载图片
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 定义棋盘格三维点
    objp = np.zeros((corner_point_long * corner_point_short, 3), np.float32)
    objp[:, :2] = np.mgrid[0:corner_point_long, 0:corner_point_short].T.reshape(-1, 2) * square_size

    # 检测角点
    ret, corners = cv2.findChessboardCorners(gray, (corner_point_long, corner_point_short), None)
    if not ret:
        print("未检测到角点！")
        return None

    # 求解棋盘格在相机坐标系下的位姿
    ret, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)
    R_obj2cam, _ = cv2.Rodrigues(rvec)

    # 构造棋盘格在相机坐标系下的齐次变换
    H_obj2cam = np.eye(4)
    H_obj2cam[:3, :3] = R_obj2cam
    H_obj2cam[:3, 3] = tvec.flatten()

    # 构造相机到基座的齐次变换
    H_cam2base = np.eye(4)
    H_cam2base[:3, :3] = R_cam2base
    H_cam2base[:3, 3] = T_cam2base.flatten()

    # 转换：棋盘格 → 相机 → 基座
    H_obj2base = H_cam2base @ H_obj2cam
    obj_in_base = H_obj2base[:3, 3]

    print("棋盘格中心在基座坐标系下的坐标 (m):", obj_in_base)

    return obj_in_base


def verify_calibration_by_realsense_camera():
    """
    通过RealSense相机验证手眼标定结果，支持用户点击图像获取3D坐标
    优化点：增加错误处理、可视化标记、深度平滑、结果保存和用户提示
    """
    # ========== 配置与常量定义 ==========
    CALIB_RESULT_DIR = "./calibrate_result"
    R_FILE = os.path.join(CALIB_RESULT_DIR, "R_cam2base.txt")
    T_FILE = os.path.join(CALIB_RESULT_DIR, "t_cam2base.txt")
    CAM_MATRIX_FILE = os.path.join(CALIB_RESULT_DIR, "camera_matrix.txt")
    DEPTH_SMOOTH_KERNEL = 3  # 深度值平滑的卷积核大小
    SAVE_RESULTS = True  # 是否保存测量结果
    RESULTS_FILE = "calibration_verification_results.txt"

    # 存储测量结果
    measurement_results = []

    # ========== 加载标定数据（带错误处理） ==========
    try:
        # 检查标定文件是否存在
        if not all(os.path.exists(f) for f in [R_FILE, T_FILE, CAM_MATRIX_FILE]):
            raise FileNotFoundError("标定结果文件不完整，请检查calibrate_result目录")

        # 读取手眼标定结果（相机->基座外参）
        R_cam2base = np.loadtxt(R_FILE)
        T_cam2base = np.loadtxt(T_FILE).reshape(3, 1)

        # 验证矩阵维度
        if R_cam2base.shape != (3, 3):
            raise ValueError(f"旋转矩阵R应为3x3，实际为{R_cam2base.shape}")
        if T_cam2base.shape != (3, 1):
            raise ValueError(f"平移向量T应为3x1，实际为{T_cam2base.shape}")

        # 读取相机内参
        mtx = np.loadtxt(CAM_MATRIX_FILE)
        print("成功加载标定数据")

    except Exception as e:
        print(f"加载标定数据失败: {str(e)}")
        return

    # ========== 初始化相机（带错误处理） ==========
    try:
        device_id = "246422072474"
        print(f"尝试连接RealSense相机 (ID: {device_id})...")

        cam = RealSenseCamera(device_id=device_id)
        cam.connect()

        # 获取相机内参（优先使用实时获取的内参）
        intr = cam.intrinsics
        fx, fy = intr.fx, intr.fy
        cx, cy = intr.ppx, intr.ppy

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

            # 1. 获取深度值（使用周围像素平均值减少噪声）
            half_kernel = DEPTH_SMOOTH_KERNEL // 2
            y_min = max(0, y - half_kernel)
            y_max = min(height, y + half_kernel + 1)
            x_min = max(0, x - half_kernel)
            x_max = min(width, x + half_kernel + 1)

            # 提取区域深度值并过滤无效值
            depth_roi = depth[y_min:y_max, x_min:x_max, 0]
            valid_depth = depth_roi[depth_roi > 0.1]  # 过滤过近或无效深度

            if len(valid_depth) == 0:
                print("所选点深度值无效，请重新选择")
                return

            depth_value = np.mean(valid_depth)  # 取平均值
            if depth_value < 0.1 or depth_value > 5.0:  # 合理深度范围判断
                print(f"深度值({depth_value:.3f}m)超出有效范围(0.1-5.0m)")
                return

            # 2. 将像素点投影到相机坐标系
            Xc = (x - cx) * depth_value / fx
            Yc = (y - cy) * depth_value / fy
            Zc = depth_value
            Pc = np.array([[Xc], [Yc], [Zc]])

            # 3. 相机坐标系 → 基座坐标系
            Pbase = R_cam2base @ Pc + T_cam2base
            Pbase = Pbase.reshape(-1)

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
            images = cam.get_image_bundle()
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
        cam.disconnect()  # 确保相机关闭
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
    images_dir = "./checkerboard_images"  # 手眼标定采集的标定版图片所在路径
    robot_tcp_pose_path = "./robot_tcp_pose.txt"  # 采集标定板图片时对应的机械臂末端的位姿 从 第一行到最后一行 需要和采集的标定板的图片顺序进行对应
    corner_point_long = 8  # 标定板角点数量  长边
    corner_point_short = 8
    corner_point_size = 0.018  # 标定板方格真实尺寸  m

    # 手眼标定 眼在手外 获取相机坐标系到机械臂基坐标系下的变换矩阵
    R_cam2base, T_cam2base, mtx, dist = compute_T(images_dir,
                                                  robot_tcp_pose_path,
                                                  corner_point_long,
                                                  corner_point_short,
                                                  corner_point_size)

    # 验证标定结果
    image_path = "D:\\PycharmProjects\\robotic-grasping\\calibrate\\checkerboard_images\\1.png"
    # verify_calibration_by_image(image_path,
    #                             R_cam2base=R_cam2base,
    #                             T_cam2base=T_cam2base,
    #                             camera_matrix=mtx,
    #                             dist_coeffs=dist,
    #                             corner_point_long=corner_point_long,
    #                             corner_point_short=corner_point_short,
    #                             square_size=corner_point_size,
    #                             )

    verify_calibration_by_realsense_camera()