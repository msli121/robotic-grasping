# -*- coding: utf-8 -*-
# @Time       : 2025/9/7 22:01
# @File       : predict_grasp.py
# @Description:
import argparse
import logging
import numpy as np

import cv2
import torch.utils.data

from hardware.camera import RealSenseCamera
from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.dataset_processing.grasp import detect_grasps

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate network')
    default_net_path = r'D:\PycharmProjects\robotic-grasping\logs\20250907_1739_training_cornell_grconvnet_goa_aff\best_iou_epoch_16_iou_0.9492'
    parser.add_argument('--network', type=str, default=default_net_path,
                        help='Path to saved network to evaluate')
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


if __name__ == '__main__':
    args = parse_args()

    # Connect to Camera
    logging.info('Connecting to camera...')
    cam = RealSenseCamera()
    cam.connect()
    cam_data = CameraData(include_depth=args.use_depth, include_rgb=args.use_rgb)

    # Load Network
    logging.info(f'Loading model... {args.network}')
    net = torch.load(args.network)
    logging.info('Done')

    # Get the compute device
    device = get_device(args.force_cpu)

    try:
        while True:
            image_bundle = cam.get_image_bundle()
            rgb_full = image_bundle['rgb_full']
            depth_full = image_bundle['aligned_depth']

            # 原图裁剪+组合
            x, depth_img, rgb_img = cam_data.get_data(rgb=rgb_full, depth=depth_full)

            with torch.no_grad():
                xc = x.to(device)
                pred = net.predict(xc)

                q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])

                gs = detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=args.n_grasps)
                if len(gs) == 0:
                    print('No grasps found')
                    continue

                # 原图
                display_img = cv2.cvtColor(rgb_full.copy(), cv2.COLOR_RGB2BGR)
                # 获取裁剪区域在原始全尺寸图像中的左上角偏移量 (y, x)
                top_left_offset = cam_data.top_left
                offset_x, offset_y = top_left_offset[1], top_left_offset[0]  # (x, y) 格式

                if not gs:
                    logging.info('No grasps found.')
                else:
                    for g in gs:
                        # Grasp 对象 g 中的 center, length, width, angle 都是相对于裁剪后的图像的
                        # 需要将这些坐标转换到原始全尺寸图像的坐标系中
                        # g.as_gr 会返回 GraspRectangle 对象，其 points 是相对于裁剪后的图像的 (y, x) 坐标
                        gr_cropped_coords = g.as_gr
                        # 1. 获取 GraspRectangle 的四个顶点坐标
                        points_cropped = gr_cropped_coords.points.astype(np.int32)
                        # 2. 将这些点加上裁剪偏移量，转换到原始全尺寸图像的坐标系
                        points_full_img = points_cropped.copy()
                        points_full_img[:, 0] += offset_y  # 加上Y偏移量
                        points_full_img[:, 1] += offset_x  # 加上X偏移量
                        # 3. 使用 cv2.polylines 绘制转换后的多边形
                        # 第三个参数 True 表示闭合多边形
                        # 颜色为红色 (BGR格式)，线宽为2
                        cv2.polylines(display_img, [points_full_img], True, (0, 0, 255), 2)
                # 显示图像
                cv2.imshow('Grasp Prediction', display_img)

                # 退出按键q 或 esc 键
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q') or key == 27:  # 27是ESC键
                    break
    finally:
        # 清理资源
        cam.disconnect()
        cv2.destroyAllWindows()
