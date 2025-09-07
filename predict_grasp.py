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
            rgb = image_bundle['rgb']
            depth = image_bundle['aligned_depth']
            x, depth_img, rgb_img = cam_data.get_data(rgb=rgb, depth=depth)

            with torch.no_grad():
                xc = x.to(device)
                pred = net.predict(xc)

                q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])

                gs = detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=args.n_grasps)
                if len(gs) == 0:
                    print('No grasps found')
                    continue

                # 使用opencv画出抓取预测框并且显示出来
                # 准备显示图像，将RGB转换为BGR格式（OpenCV默认格式）
                if args.use_rgb:
                    rgb_img = cam_data.get_rgb(rgb, False)
                    display_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
                else:
                    depth_img = np.squeeze(cam_data.get_depth(depth))
                    # 如果不使用RGB，则使用深度图并转换为3通道以便绘制彩色框
                    display_img = cv2.cvtColor((depth_img / np.max(depth) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

                # 绘制每个抓取预测框
                for g in gs:
                    # 获取GraspRectangle对象
                    gr = g.as_gr
                    # 获取四个顶点坐标并转换为整数
                    points = gr.points.astype(np.int32)
                    # 调整坐标格式以适应OpenCV的polylines函数
                    pts = points.reshape((-1, 1, 2))
                    # 绘制多边形（抓取框）- 蓝色
                    cv2.polylines(display_img, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
                    # 在中心点绘制一个小圆 - 红色
                    center_np = np.array(g.center)  # tuple → numpy 数组
                    center = tuple(center_np.astype(np.int32)[::-1])  # 转int32 → 反转 → 转tuple
                    cv2.circle(display_img, center, 5, (0, 0, 255), -1)
                    # 显示抓取角度
                    angle_text = f"angle:{g.angle * 180 / np.pi:.1f}"
                    cv2.putText(display_img, angle_text, (center[0] + 10, center[1] + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

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
