import logging
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs
from matplotlib import cm
from scipy import ndimage
from sklearn.neighbors import NearestNeighbors

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RealSenseCamera:
    def __init__(self,
                 device_id=None,
                 width=640,
                 height=480,
                 fps=6):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps

        self.pipeline = None
        self.scale = None
        self.intrinsics = None
        self.K = None
        self.dist = None

    def connect(self):
        # Start and configure
        self.pipeline = rs.pipeline()
        config = rs.config()
        if self.device_id:
            config.enable_device(str(self.device_id))
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        cfg = self.pipeline.start(config)

        # Determine intrinsics
        rgb_profile = cfg.get_stream(rs.stream.color)
        self.intrinsics = rgb_profile.as_video_stream_profile().get_intrinsics()

        # 构造内参矩阵
        self.K = np.array([[self.intrinsics.fx, 0, self.intrinsics.ppx],
                           [0, self.intrinsics.fy, self.intrinsics.ppy],
                           [0, 0, 1]])
        # 构造畸变系数矩阵
        self.dist = np.array([self.intrinsics.coeffs[0], self.intrinsics.coeffs[1], self.intrinsics.coeffs[2],
                              self.intrinsics.coeffs[3], self.intrinsics.coeffs[4]])

        # Determine depth scale
        self.scale = cfg.get_device().first_depth_sensor().get_depth_scale()

    def disconnect(self):
        if self.pipeline:
            self.pipeline.stop()
            self.pipeline = None

    def get_image_bundle(self, fill_depth=True, fill_method='opencv'):
        """
        获取图像帧
        :param fill_depth: 是否填充深度图缺失值
        :param fill_method: 填充方法，可选 opencv bilateral weighted median
        :return:
                RGB图像 rgb: np.ndarray [H, W, 3],
                对齐后的深度图像 depth: np.ndarray [H, W, 1]
        """
        frames = self.pipeline.wait_for_frames()

        align = rs.align(rs.stream.color)
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.first(rs.stream.color)
        color_image = np.asanyarray(color_frame.get_data())

        aligned_depth_frame = aligned_frames.get_depth_frame()
        depth_image = np.asarray(aligned_depth_frame.get_data(), dtype=np.float32)
        # 转换为米
        depth_image *= self.scale
        if fill_depth:
            # 将0值（无效深度）转换为NaN
            depth_image[depth_image <= 0] = np.nan
            if fill_method == 'opencv':
                depth_image = self._fill_depth_opencv(depth_image)
            elif fill_method == 'bilateral':
                depth_image = self._fill_depth_bilateral(depth_image)
            elif fill_method == 'weighted':
                depth_image = self._fill_depth_weighted(depth_image)
            elif fill_method == 'median':
                depth_image = self._fill_depth_median(depth_image)
            else:
                raise ValueError(f"Unknown fill_method: {fill_method}")
        # 扩展通道维度 变为[H,W,1]
        depth_image = np.expand_dims(depth_image, axis=2)

        return {
            'rgb': color_image,
            'aligned_depth': depth_image,
        }

    def plot_image_bundle(self):
        images = self.get_image_bundle()

        rgb = images['rgb']
        depth = images['aligned_depth']

        fig, ax = plt.subplots(1, 2, squeeze=False)
        ax[0, 0].imshow(rgb)
        m, s = np.nanmean(depth), np.nanstd(depth)
        ax[0, 1].imshow(depth.squeeze(axis=2), vmin=m - s, vmax=m + s, cmap=plt.cm.gray)
        ax[0, 0].set_title('rgb')
        ax[0, 1].set_title('aligned_depth')

        plt.show()

    def get_K_and_dist(self):
        """
        获取相机内参矩阵
        :return: 内参矩阵K
        """
        return self.K, self.dist

    def set_K(self, K):
        self.K = K

    def set_dist(self, dist):
        self.dist = dist

    def _fill_depth_opencv(self, depth_map: np.ndarray) -> np.ndarray:
        """
        使用OpenCV的Telea算法填充深度图中的缺失值

        参数:
            depth_map: 包含NaN值的深度图(单通道)

        返回:
            填充后的深度图
        """
        start_time = time.time()

        # 1. 创建缺失区域掩码 (缺失区域为255，有效区域为0)
        nan_mask = np.isnan(depth_map).astype(np.uint8) * 255

        # 2. 对掩码进行形态学处理，去除噪点并连接区域
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(nan_mask, cv2.MORPH_CLOSE, kernel)

        # 3. 保存原始深度图的统计信息，用于后续归一化
        valid_depth = depth_map[~np.isnan(depth_map)]
        if len(valid_depth) == 0:
            raise ValueError("输入深度图中没有有效深度值")

        min_val, max_val = valid_depth.min(), valid_depth.max()

        # 4. 将深度图归一化到0-255范围(OpenCV inpaint需要)
        depth_normalized = cv2.normalize(
            depth_map, None, 0, 255,
            cv2.NORM_MINMAX, dtype=cv2.CV_8U,
            mask=~np.isnan(depth_map).astype(np.uint8)
        )
        depth_normalized[np.isnan(depth_map)] = 0  # 缺失区域设为0

        # 5. 应用Telea快速修复算法
        # 参数说明:
        # - 3: 修复邻域半径
        # - cv2.INPAINT_TELEA: 快速行进算法，速度快且效果好
        filled_normalized = cv2.inpaint(depth_normalized, mask, 3, cv2.INPAINT_TELEA)

        # 6. 将填充结果恢复到原始深度范围
        filled_depth = cv2.normalize(
            filled_normalized, None,
            min_val, max_val,
            cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )

        # 7. 只替换原始缺失的区域，保留有效深度值
        result = np.where(np.isnan(depth_map), filled_depth, depth_map)

        logger.debug(f"opencv 深度图填充耗时: {time.time() - start_time:.4f}秒")
        return result

    def _fill_depth_bilateral(self, depth_map: np.ndarray, iterations: int = 3) -> np.ndarray:
        """
        使用双边滤波填充深度图缺失值（保留边缘的同时填充，适合需要保持物体轮廓的场景）

        :param depth_map: 包含NaN的深度图
        :param iterations: 迭代次数
        :return: 填充后的深度图
        """
        start_time = time.time()
        filled = depth_map.copy()

        for _ in range(iterations):
            # 找到NaN区域
            nan_mask = np.isnan(filled)

            if not np.any(nan_mask):
                break

            # 使用双边滤波进行填充，保留边缘
            filled[nan_mask] = ndimage.generic_filter(
                filled,
                lambda x: np.nanmean(x),
                size=5,
                mode='nearest'
            )[nan_mask]

        logger.debug(f"双边滤波填充耗时: {time.time() - start_time:.4f}秒")
        return filled

    def _fill_depth_weighted(self, depth_map: np.ndarray, radius: int = 3) -> np.ndarray:
        """
        使用邻近有效像素的加权平均，填充结果更平滑

        :param depth_map: 包含NaN的深度图
        :param radius: 搜索邻域半径
        :return: 填充后的深度图
        """
        start_time = time.time()
        filled = depth_map.copy()
        nan_mask = np.isnan(filled)

        if not np.any(nan_mask):
            return filled

        # 获取NaN点坐标
        nan_coords = np.argwhere(nan_mask)
        # 获取有效点坐标和值
        valid_coords = np.argwhere(~nan_mask)
        valid_values = filled[~nan_mask]

        # 使用KNN找到最近的有效点
        nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(valid_coords)
        distances, indices = nbrs.kneighbors(nan_coords)

        # 基于距离的加权平均
        weights = 1.0 / (distances + 1e-8)  # 避免除零
        normalized_weights = weights / np.sum(weights, axis=1, keepdims=True)

        # 计算填充值
        filled_values = np.sum(normalized_weights * valid_values[indices], axis=1)

        # 填充NaN值
        filled[tuple(nan_coords.T)] = filled_values

        logger.debug(f"加权平均填充耗时: {time.time() - start_time:.4f}秒")
        return filled

    def _fill_depth_median(self, depth_map: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        使用中值滤波快速填充深度图缺失值，适合实时应用场景

        :param depth_map: 包含NaN的深度图
        :param kernel_size: 中值滤波核大小
        :return: 填充后的深度图
        """
        start_time = time.time()
        # 将NaN替换为0以便进行中值滤波
        depth_with_zero = np.nan_to_num(depth_map, nan=0)

        # 执行中值滤波
        median_filtered = ndimage.median_filter(depth_with_zero, size=kernel_size)

        # 只替换原来的NaN区域
        filled = np.where(np.isnan(depth_map), median_filtered, depth_map)

        logger.debug(f"中值滤波填充耗时: {time.time() - start_time:.4f}秒")
        return filled


def test_depth_fill_methods(depth_npy_path):
    """
    测试并可视化四种深度图填充方法的效果
    :param depth_npy_path: 深度图npy文件路径（如/xxx/00_depth_raw.npy）
    """

    cam = RealSenseCamera(device_id=0)  # device_id可任意填写

    # 1. 加载深度数据并预处理
    depth_data = np.load(depth_npy_path)  # 形状为[H, W, 1]
    depth_2d = np.squeeze(depth_data)  # 转为[H, W]
    logger.info(f"加载深度图: {depth_npy_path}, 形状: {depth_2d.shape}")

    # 模拟原始数据中的无效值（0值转为NaN，与相机输出一致）
    depth_original = depth_2d.copy()
    depth_original[depth_original <= 0] = np.nan  # 无效深度设为NaN

    # 2. 准备填充方法列表
    fill_methods = [
        ("Origin", None),  # 原图不填充
        ("OpenCV Telea", cam._fill_depth_opencv),
        ("bilateral", cam._fill_depth_bilateral),
        ("weighted", cam._fill_depth_weighted),
        ("median", cam._fill_depth_median)
    ]

    # 3. 对每种方法进行填充处理
    results = []
    for name, method in fill_methods:
        if method is None:
            # 原始图直接添加
            results.append((name, depth_original))
        else:
            # 复制数据避免修改原图
            depth_copy = depth_original.copy()
            # 调用填充方法
            filled = method(depth_copy)
            results.append((name, filled))

    # 4. 计算统一的显示范围（基于原始有效深度值）
    valid_depth = depth_original[~np.isnan(depth_original)]
    if len(valid_depth) == 0:
        logger.error("深度图中无有效数据，无法计算显示范围")
        return
    vmin, vmax = np.percentile(valid_depth, [5, 95])  # 去除极端值影响

    # 5. 可视化对比
    plt.figure(figsize=(20, 4))  # 宽屏布局，1行5列
    for i, (name, depth_img) in enumerate(results):
        plt.subplot(1, 5, i + 1)
        # 显示深度图（用jet颜色映射增强对比度）
        im = plt.imshow(depth_img, cmap=cm.jet, vmin=vmin, vmax=vmax)
        plt.title(name, fontsize=10)
        plt.axis('off')  # 关闭坐标轴

    # 添加共用颜色条
    cbar_ax = plt.gcf().add_axes([0.92, 0.15, 0.01, 0.7])  # 位置[左,下,宽,高]
    plt.colorbar(im, cax=cbar_ax, label='depth/m')

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # 预留颜色条位置
    plt.suptitle('depth fill methods', y=1.02, fontsize=12)
    plt.show()


if __name__ == '__main__':
    # cam = RealSenseCamera(device_id=246422072474)
    # cam.connect()
    # K, dist = cam.get_K_and_dist()
    # print("内参矩阵:", K)
    # print("畸变系数:", dist)
    # while True:
    #     cam.plot_image_bundle()

    depth_data_file = "/Users/a123/PycharmProjects/robotic-grasping/calibrate/data/20250819001632/00_depth_raw.npy"
    test_depth_fill_methods(depth_data_file)
