# -*- coding: utf-8 -*-
# @Time       : 2025/9/7 14:36
# @File       : utils.py.py
# @Description: 数据集预处理，重命令，数据增强
import glob
import json
import os
import random
import shutil

import albumentations as A
import cv2


# 重命名对指定文件夹下的images和labels
def rename_images_and_labels(path_dir):
    """
    对指定文件夹下的 'images' 文件夹中的图片文件重命名，并同步更新 'labels' 文件夹中对应的标签文件。
    图片重命名格式为 'imageXXX.png'，其中 XXX 为从 001 开始的三位数字。

    参数:
        path_dir (str): 包含 'images' 和 'labels' 子文件夹的根目录路径。
    """
    # 获取 'images' 和 'labels' 文件夹的路径
    images_dir = os.path.join(path_dir, 'images')
    labels_dir = os.path.join(path_dir, 'labels')

    # 检查 'images' 文件夹是否存在
    if not os.path.exists(images_dir):
        print(f"目录 '{images_dir}' 不存在，请检查路径。")
        return

    # 获取 'images' 文件夹中所有图片文件，并按名称排序
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()

    # 遍历图片文件并重命名
    for i, old_image_name in enumerate(image_files, start=1):
        # 按格式生成新的图片文件名，例如 'image001.png'
        new_image_name = f"image{str(i).zfill(3)}.png"
        old_image_path = os.path.join(images_dir, old_image_name)
        new_image_path = os.path.join(images_dir, new_image_name)

        # 重命名图片文件
        os.rename(old_image_path, new_image_path)
        print(f"图片已重命名: {old_image_name} -> {new_image_name}")

        # 如果存在 'labels' 文件夹，处理对应的标签文件
        if os.path.exists(labels_dir):
            # 推导对应标签文件的旧名称和新名称
            old_label_name = os.path.splitext(old_image_name)[0] + '.txt'
            old_label_path = os.path.join(labels_dir, old_label_name)
            # 如果标签文件存在，则进行重命名
            if os.path.exists(old_label_path):
                new_label_name = os.path.splitext(new_image_name)[0] + '.txt'
                new_label_path = os.path.join(labels_dir, new_label_name)
                os.rename(old_label_path, new_label_path)
                print(f"标签文件已重命名: {old_label_name} -> {new_label_name}")


def load_yolo_labels(label_path):
    """
    加载 YOLO 格式的标签
    Args:
        label_path (str): 标签文件路径
    Returns:
        list: 标签列表，每个元素为 (class_id, x, y, w, h)
    """
    with open(label_path, "r", encoding='utf-8') as f:
        labels = [line.strip().split() for line in f.readlines()]
    return [(int(label[0]), float(label[1]), float(label[2]), float(label[3]), float(label[4])) for label in labels]


def save_yolo_labels(output_path, labels):
    """
    保存 YOLO 格式的标签
    Args:
        output_path (str): 输出标签文件路径
        labels (list): 标签列表，每个元素为 (class_id, x, y, w, h)
    """
    with open(output_path, "w", encoding='utf-8') as f:
        for label in labels:
            f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")


def augment_dataset(image_dir, label_dir, output_image_dir, output_label_dir, num_augments=3, save_origin=True):
    """
    对数据集中的所有图片和标签进行数据增强
    Args:
        image_dir (str): 原始图片路径
        label_dir (str): 原始标签路径
        output_image_dir (str): 增强后图片保存路径
        output_label_dir (str): 增强后标签保存路径
        num_augments (int): 每张图片生成增强图片的数量
        save_origin: 是否保存原始图片和标签
    """
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
    if not os.path.exists(output_label_dir):
        os.makedirs(output_label_dir)
    total_image_count = 0
    # 定义增强操作
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),  # 水平翻转
        A.VerticalFlip(p=0.5),  # 垂直翻转
        A.RandomBrightnessContrast(p=0.5),  # 随机亮度对比度
        A.GaussNoise(p=0.3),  # 添加噪声
        A.Resize(480, 640),  # 重新调整大小
        A.Rotate(limit=30, p=0.5),  # 随机旋转
        A.RandomScale(scale_limit=0.2, p=0.5),  # 随机缩放
        A.Perspective(scale=(0.05, 0.1), p=0.5),  # 透视变换
        A.MotionBlur(blur_limit=3, p=0.3),  # 模糊
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    # 遍历所有图片
    for image_name in os.listdir(image_dir):
        if not image_name.endswith(('.jpg', '.png', '.jpeg')):
            continue
        image_path = os.path.join(image_dir, image_name)
        label_path = os.path.join(label_dir, os.path.splitext(image_name)[0] + ".txt")

        if not os.path.exists(label_path):
            print(f"标签文件不存在: {label_path}")
            continue

        # 保留原图
        if save_origin:
            output_img_path = os.path.join(output_image_dir, image_name)
            output_label_path = os.path.join(output_label_dir, os.path.splitext(image_name)[0] + ".txt")
            shutil.copy(image_path, output_img_path)
            shutil.copy(label_path, output_label_path)
            total_image_count += 1

        # 读取图片和标签
        image = cv2.imread(image_path)
        labels = load_yolo_labels(label_path)

        # 生成增强图片和标签
        for i in range(num_augments):
            augmented_image, augmented_labels = augment_image_and_labels(image, labels, transform)
            # 保存增强后的图片和标签
            new_image_name = os.path.splitext(image_name)[0] + f"_aug_{i}.jpg"
            new_label_name = os.path.splitext(image_name)[0] + f"_aug_{i}.txt"
            new_image_path = os.path.join(output_image_dir, new_image_name)
            new_label_path = os.path.join(output_label_dir, new_label_name)
            cv2.imwrite(new_image_path, augmented_image)
            save_yolo_labels(new_label_path, augmented_labels)
            total_image_count += 1

    print(f"数据增强完成，共计 {total_image_count} 张图片")
    print(f"增强后的图片保存在 {output_image_dir}")
    print(f"增强后的标签保存在 {output_label_dir}")


def split_dataset(input_dir, output_dir, train_ratio=0.8, val_ratio=0.2):
    """
    将 YOLO 数据集按指定比例分割为训练集和验证集。
    Args:
        input_dir (str): 输入的原始数据集路径，包含 'images' 和 'labels' 文件夹。
        output_dir (str): 输出文件夹路径，用于存放分割后的数据集。
        train_ratio (float): 训练集比例 (默认 0.8)。
        val_ratio (float): 验证集比例 (默认 0.2)。
    """
    # 获取 images 文件 ('.jpg', '.png', '.jpeg')
    image_files = []
    for ext in ['.jpg', '.png', '.jpeg']:
        image_files += glob.glob(os.path.join(input_dir, f"*{ext}"))
    # 获取 labels 文件
    label_files = []
    for image_file in image_files:
        label_file = os.path.join(os.path.dirname(image_file),
                                  os.path.splitext(os.path.basename(image_file))[0] + ".txt")
        if os.path.exists(label_file):
            label_files.append(label_file)

    if len(image_files) != len(label_files):
        print(f"图片文件数量 {len(image_files)} 与标签文件数量 {len(label_files)} 不一致，请检查数据集。")
        return

    # 创建训练集和验证集的文件夹
    train_images_dir = os.path.join(output_dir, 'train', 'images')
    train_labels_dir = os.path.join(output_dir, 'train', 'labels')
    val_images_dir = os.path.join(output_dir, 'val', 'images')
    val_labels_dir = os.path.join(output_dir, 'val', 'labels')
    for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # 删除目标文件夹中已有的文件
    clear_existing_files(train_images_dir)
    clear_existing_files(train_labels_dir)
    clear_existing_files(val_images_dir)
    clear_existing_files(val_labels_dir)

    # 随机打乱图片文件
    random.shuffle(image_files)

    # 根据比例计算分割点
    total_files = len(image_files)
    train_size = int(total_files * train_ratio)

    # 将图片和标签分配到训练集和验证集
    train_files = image_files[:train_size]
    val_files = image_files[train_size:]

    label_dir = os.path.dirname(label_files[0])
    # 处理训练集文件
    for image_path in train_files:
        image_base_name = os.path.basename(image_path)
        filename = os.path.splitext(image_base_name)[0]
        label_path = os.path.join(label_dir, filename + ".txt")

        # 复制图片和标签到训练集文件夹
        shutil.copy(image_path, os.path.join(train_images_dir, image_base_name))
        shutil.copy(label_path, os.path.join(train_labels_dir, filename + ".txt"))

    # 处理验证集文件
    for image_path in val_files:
        image_base_name = os.path.basename(image_path)
        filename = os.path.splitext(image_base_name)[0]
        label_path = os.path.join(label_dir, filename + ".txt")

        # 复制图片和标签到验证集文件夹
        shutil.copy(image_path, os.path.join(val_images_dir, image_base_name))
        shutil.copy(label_path, os.path.join(val_labels_dir, filename + ".txt"))

    print(f"数据集分割完成: 训练集({len(train_files)} 张图片)和验证集({len(val_files)} 张图片)")


def batch_convert_json_to_txt(input_json_dir, output_txt_dir, class_txt_path):
    """
    批量将 JSON 格式的标签转换为 TXT 格式
    Args:
        input_json_dir (str): 输入 JSON 文件夹路径
        output_txt_dir (str): 输出 TXT 文件夹路径
        class_txt_path (str): 类别名称文件路径
    """
    if not os.path.exists(input_json_dir):
        print(f"输入目录不存在: {input_json_dir}")
        return
    if not os.path.exists(output_txt_dir):
        os.makedirs(output_txt_dir)
    if not os.path.exists(class_txt_path):
        print(f"类别文件不存在: {class_txt_path}")
        return
    # 读取类别名称，跳过空白行
    with open(class_txt_path, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]
    class_map = {name: i for i, name in enumerate(class_names)}

    json_files = glob.glob(os.path.join(input_json_dir, '*.json'))
    json_files.sort()
    for json_file in json_files:
        basename = os.path.basename(json_file)
        # 构建对应的输出 TXT 文件路径
        output_txt_file = os.path.join(output_txt_dir, basename.replace('.json', '.txt'))
        # 调用 yolo_to_json 函数进行转换
        json_to_yolo(json_file, output_txt_file, class_map)
    print(f"所有 JSON 文件已转换为 YOLOv8 格式，个数：{len(json_files)}，保存至: {output_txt_dir}")


def augment_image_and_labels(image, labels, transform):
    """
    对图像和标签进行数据增强
    Args:
        image (ndarray): 原始图像
        labels (list): YOLO 格式标签 (class_id, x, y, w, h)
        transform (albumentations.Compose): 数据增强操作
    Returns:
        ndarray: 增强后的图像
        list: 增强后的标签
    """
    # YOLO 格式坐标无需转换为像素值，直接作为归一化坐标传递给 Albumentations
    bboxes = [
        [label[1], label[2], label[3], label[4], label[0]]  # YOLO 格式：x_center, y_center, width, height, class_id
        for label in labels
    ]

    # 进行增强
    transformed = transform(image=image, bboxes=bboxes, class_labels=[label[0] for label in labels])
    augmented_image = transformed['image']
    augmented_bboxes = transformed['bboxes']
    # 转换增强后的标签回 YOLO 格式
    augmented_labels = [
        (bbox[4], bbox[0], bbox[1], bbox[2], bbox[3])  # class_id, x_center, y_center, width, height
        for bbox in augmented_bboxes
    ]
    return augmented_image, augmented_labels


def clear_existing_files(directory):
    """
    清空给定目录中的所有文件（图片和标签）。
    Args:
        directory (str): 目标目录路径，包含图片和标签文件。
    """
    if os.path.exists(directory):
        for root, dirs, files in os.walk(directory, topdown=False):
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)  # 删除文件
    else:
        print(f"目录 '{directory}' 不存在，跳过删除操作。")


def yolo_to_json(txt_file_path, json_save_dir, image_path, class_map):
    """
    将 YOLOv8 格式的标签（TXT）转换为 JSON 格式。
    Args:
        txt_file_path (str): YOLOv8 标签文件路径。
        json_save_dir (str): 输出 JSON 保存文件夹
        image_path (str): 图片文件路径
        image_height (int): 图片高度。
        class_map (dict): 类别 ID 到标签名的映射，例如 {9: "圆柱体"}。
    """
    # 获取图片宽高
    image = cv2.imread(image_path)
    image_height, image_width, _ = image.shape
    shapes = []
    with open(txt_file_path, "r", encoding='utf-8') as f:
        for line in f.readlines():
            class_id, x_center, y_center, box_width, box_height = map(float, line.strip().split())
            class_id = int(class_id)
            label = class_map.get(class_id, f"class_{class_id}")

            # 计算左上角和右下角的像素坐标
            x1 = (x_center - box_width / 2) * image_width
            y1 = (y_center - box_height / 2) * image_height
            x2 = (x_center + box_width / 2) * image_width
            y2 = (y_center + box_height / 2) * image_height

            # 构造 JSON 形状
            shape = {
                "label": label,
                "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                "group_id": None,
                "description": "",
                "difficult": False,
                "shape_type": "rectangle",
                "flags": {},
                "attributes": {}
            }
            shapes.append(shape)

    # 构造完整 JSON 数据
    json_data = {
        "version": "2.3.6",
        "flags": {},
        "shapes": shapes,
        "imagePath": txt_file_path.replace(".txt", ".jpg"),
        "imageData": None,
        "imageHeight": image_height,
        "imageWidth": image_width
    }
    # 确认json文件路径
    json_path = os.path.join(json_save_dir, os.path.splitext(os.path.basename(txt_file_path))[0] + ".json")
    # 保存到 JSON 文件
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)


def json_to_yolo(json_file: str, txt_file: str, label_map: dict):
    """
    将 JSON 格式的标签转换为 YOLOv8 格式。
    Args:
        json_file (str): 输入 JSON 文件路径。
        txt_file (str): 输出 YOLOv8 标签文件路径。
        label_map (dict): 标签名到类别 ID 的映射，例如 {"圆柱体": 9}。
    """
    with open(json_file, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    image_width = json_data.get('imageWidth', 640)
    image_height = json_data.get('imageHeight', 480)

    lines = []
    for shape in json_data.get("shapes", []):
        label = shape.get('label')
        if not label:
            print(f"{json_file} 未找到标签")
            continue
        class_id = label_map.get(label, -1)
        if class_id == -1:
            continue  # 如果标签未映射，跳过

        # 获取矩形框的四个点
        points = shape["points"]
        x1, y1 = points[0]
        x2, y2 = points[2]

        # 计算中心点和宽高的归一化值
        x_center = ((x1 + x2) / 2) / image_width
        y_center = ((y1 + y2) / 2) / image_height
        box_width = (x2 - x1) / image_width
        box_height = (y2 - y1) / image_height

        # 构造 YOLO 格式
        line = f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        lines.append(line)

    # 保存到 TXT 文件
    with open(txt_file, "w", encoding='utf-8') as f:
        f.write("\n".join(lines))


def replace_class_name(label_dir):
    """
    替换标签目录下所有文件中的类名
    Args:
        label_dir (str): 标签目录路径
    """
    if not os.path.exists(label_dir):
        print(f"输入目录不存在: {label_dir}")
        return
    # 获取指定目录下所有的json文件
    json_files = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.json')]
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        for shape in json_data['shapes']:
            shape['label'] = shape['label'].replace('直角工具刀', '内六角扳手')
            shape['label'] = shape['label'].replace('多功能小刀', '折叠小刀')
            shape['label'] = shape['label'].replace('美工刀', '水果刀')
            shape['label'] = shape['label'].replace('开关配件', '电子开关')
            shape['label'] = shape['label'].replace('涂卡笔', '2B铅笔')
            shape['label'] = shape['label'].replace('夹子', '木夹')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    # 对应原始自制数据集进行重命名
    # rename_images_and_labels('./data_origin')

    # 对数据集进行增强
    # augment_dataset('./data_origin/images', './data_origin/labels', './data_augmented/images',
    #                 './data_augmented/labels',
    #                 num_augments=3)

    # 替换类名
    label_dir = r'D:\datasets\paper_dataset\rgb'
    replace_class_name(label_dir)

    # 将JSON格式转为txt
    input_json_dir = r'D:\datasets\paper_dataset\rgb'
    output_txt_dir = r'D:\datasets\paper_dataset\rgb'
    class_txt_path = r'D:\datasets\paper_dataset\classes_zh.txt'
    batch_convert_json_to_txt(input_json_dir, output_txt_dir, class_txt_path)

    # 数据分割
    input_dir = r'D:\datasets\paper_dataset\rgb'
    output_dir = r'D:\PycharmProjects\robotic-grasping\yolo\datasets\paper'
    split_dataset(input_dir, output_dir, train_ratio=0.8, val_ratio=0.2)

    # 将 YOLOv8 格式的标签转换为 JSON 格式
    # txt_path = './data_augmented/labels/image001_aug_0.txt'
    # json_save_dir = './data_augmented/images'
    # image_path = './data_augmented/images/image001_aug_0.jpg'
    # class_map = {7: "圆柱体"}
    # yolo_to_json(txt_path, json_save_dir, image_path, class_map)
