# -*- coding: utf-8 -*-
# @Time       : 2025/9/7 22:01
# @File       : grasp_predictor.py
# @Description:
import argparse
import logging
import os

import numpy as np

import cv2
import torch.utils.data

from hardware.camera import RealSenseCamera
from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.dataset_processing.grasp import detect_grasps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraspPredictor(object):
    def __init__(self, model_path, output_size=224):
        super().__init__()
        self.model = None
        self.device = None
        self.model_path = model_path
        self.output_size = output_size

    def load_model(self):
        logger.info(f"Loading grasp predictor model from: {self.model_path}")
        if  os.path.exists(self.model_path):
            self.model = torch.load(self.model_path)
            logger.info("Grasp predictor model loaded successfully.")
            self.device = get_device(False)
        else:
            logger.error(f"Grasp model file {self.model_path} does not exist")

    def predict(self, rgb, depth, no_grasps=1):
        """
        :param rgb: RGB image, np.ndarray [H,W,3]
        :param depth: Depth image [H,W,1]
        :param no_grasps: Number of grasps to detect
        :return:
        """
        rgb_shape = rgb.shape
        cam_data = CameraData(width=rgb_shape[1], height=rgb_shape[0],
                              output_size=self.output_size,
                              include_rgb=True,
                              include_depth=True)
        # 原图裁剪+组合
        x, depth_img, rgb_img = cam_data.get_data(rgb=rgb, depth=depth)
        # 模型预测抓取
        with torch.no_grad():
            xc = x.to(self.device)
            pred = self.model.predict(xc)
            q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])
            gs = detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=no_grasps)
            # 获取裁剪区域在原始全尺寸图像中的左上角偏移量 (y, x)
            top_left_offset = cam_data.top_left
            offset_x, offset_y = top_left_offset[1], top_left_offset[0]  # (x, y) 格式
            grasp_infos = []
            if not gs:
                logging.info('No grasps found.')
            else:
                for g in gs:
                    # Grasp 对象 g 中的 center, length, width, angle 都是相对于裁剪后的图像的
                    # 相对于裁剪后的图像的 (y, x) 坐标
                    gr_cropped_yx = g.as_gr
                    gr_cropped_yx = gr_cropped_yx.points.astype(np.int32)
                    # 获取四个顶点坐标, 并转换为 (x, y) 格式
                    points_cropped_xy = np.array([[p[1], p[0]] for p in gr_cropped_yx])
                    # 将这些点加上裁剪偏移量，转换到原始全尺寸图像的坐标系
                    points_cropped_xy[:, 0] += offset_x  # 加上X偏移量
                    points_cropped_xy[:, 1] += offset_y  # 加上Y偏移量
                    grasp_info = {
                        'angle': g.angle,
                        'width': g.width,
                        'length': g.length,
                        'center': list(g.center),
                        'points': points_cropped_xy,
                    }
                    grasp_infos.append(grasp_info)
            return grasp_infos, q_img, ang_img, width_img


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate network')
    default_net_path = r'D:\PycharmProjects\robotic-grasping\trained-models\cornell-randsplit-rgbd-grconvnet3-drop1-ch32\epoch_19_iou_0.98'
    # default_net_path = r'D:\PycharmProjects\robotic-grasping\trained-pretrained_models\jacquard-rgbd-grconvnet3-drop0-ch32\epoch_48_iou_0.93'
    parser.add_argument('--network', type=str, default=default_net_path,
                        help='Path to saved network to evaluate')
    parser.add_argument('--input-size', type=int, default=224,
                        help='Input image size for the network')
    parser.add_argument('--use-depth', type=int, default=1,
                        help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1,
                        help='Use RGB image for evaluation (1/0)')
    parser.add_argument('--n-grasps', type=int, default=1,
                        help='Number of grasps to consider per image')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Connect to Camera
    logging.info('Connecting to camera...')
    cam = RealSenseCamera()
    cam.connect()

    # Load Network
    logging.info(f'Loading model... {args.network}')
    predictor = GraspPredictor(args.network, output_size=args.input_size)
    predictor.load_model()
    logging.info('Done')

    try:
        while True:
            image_bundle = cam.get_image_bundle()
            rgb = image_bundle['rgb']
            depth = image_bundle['aligned_depth']

            # 预测
            grasps, _, _, _ = predictor.predict(rgb, depth)
            # 原图
            display_img = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
            if not grasps:
                logging.info('No grasps found.')
            else:
                for grasp in grasps:
                    # 使用 cv2.polylines 绘制转换后的多边形
                    cv2.polylines(display_img, [grasp['points']], True, (0, 0, 255), 1)
            # 显示图像
            cv2.imshow('Grasp Prediction', display_img)
            # 暂停0.1秒，让图片显示
            cv2.waitKey(100)

            # 退出按键q 或 esc 键
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:  # 27是ESC键
                break
    finally:
        # 清理资源
        cam.disconnect()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
