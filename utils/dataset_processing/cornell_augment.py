# # utils/cornell_augment.py
# import os
# import random
# import cv2
# import numpy as np
# from tqdm import tqdm
# from utils.dataset_processing import grasp
#
# def generate_cornell_augmented(input_dir, output_dir, output_size=224, total_samples=51000):
#     """
#     离线生成增强版 Cornell 数据集（论文方式）
#     :param input_dir: 原始 Cornell 数据集路径
#     :param output_dir: 增强后保存路径
#     :param output_size: 裁剪后大小
#     :param total_samples: 生成总样本数
#     """
#     os.makedirs(output_dir, exist_ok=True)
#     os.makedirs(os.path.join(output_dir, 'rgb'), exist_ok=True)
#     os.makedirs(os.path.join(output_dir, 'depth'), exist_ok=True)
#     os.makedirs(os.path.join(output_dir, 'pos_label'), exist_ok=True)
#
#     # 收集原始 RGB 文件列表
#     rgb_files = [f for f in os.listdir(input_dir) if f.endswith('r.png')]
#
#     for i in tqdm(range(total_samples), desc="Generating Augmented Cornell"):
#         rgb_path = os.path.join(input_dir, random.choice(rgb_files))
#         base = rgb_path[:-6]  # 去掉 "_r.png"
#         depth_path = base + 'd.tiff'
#         pos_path = base + 'cpos.txt'
#
#         # 读取数据
#         rgb_img = cv2.imread(rgb_path)
#         depth_img = cv2.imread(depth_path, -1).astype(np.float32)
#         depth_img = np.expand_dims(depth_img, axis=2)  # H,W,1
#
#         # 读取正抓取框
#         gtbbs = grasp.GraspRectangles.load_from_cornell_file(pos_path)
#
#         # 随机旋转
#         rot = random.uniform(-np.pi/2, np.pi/2)
#         rgb_img = grasp.rotate(rgb_img, rot)
#         depth_img = grasp.rotate(depth_img, rot)
#         gtbbs.rotate(rot, rgb_img.shape[1]//2, rgb_img.shape[0]//2)
#
#         # 随机缩放
#         zoom_factor = random.uniform(0.9, 1.1)
#         rgb_img = grasp.zoom(rgb_img, zoom_factor)
#         depth_img = grasp.zoom(depth_img, zoom_factor)
#         gtbbs.zoom(zoom_factor)
#
#         # 随机裁剪到 output_size
#         rgb_img, depth_img, gtbbs = grasp.random_crop(rgb_img, depth_img, gtbbs, output_size)
#
#         # 生成 pos, cos, sin, width 标签
#         pos_img, cos_img, sin_img, width_img = gtbbs.draw(output_size)
#
#         # 保存
#         cv2.imwrite(os.path.join(output_dir, 'rgb', f'{i:05d}.png'), rgb_img)
#         np.save(os.path.join(output_dir, 'depth', f'{i:05d}.npy'), depth_img)
#         np.savez_compressed(os.path.join(output_dir, 'pos_label', f'{i:05d}.npz'),
#                             pos=pos_img, cos=cos_img, sin=sin_img, width=width_img)
#
#     print(f"增强数据已生成到 {output_dir}，共 {total_samples} 样本。")
