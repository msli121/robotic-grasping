import os
import argparse
import json
import time

from ultralytics import YOLO
import matplotlib.pyplot as plt

# --- Matplotlib 设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 项目根目录定义 ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Train YOLOv8 on a custom dataset.")
    # model_path = r'D:\PycharmProjects\robotic-grasping\yolov8\models\yolov8s.pt'
    model_path = r'D:\PycharmProjects\robotic-grasping\yolo\pretrained_models\yolo11s.pt'
    parser.add_argument('--model-cfg', type=str, default=model_path,
                        help='Starting model configuration, e.g., yolov8n.pt')
    parser.add_argument('--data-cfg', type=str, default=r'datasets/paper/dataset.yaml',
                        help='Path to dataset.yaml relative to the ROOT_DIR')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--workers', type=int, default=0, help='Dataloader workers (0 for Windows is recommended)')
    parser.add_argument('--name', type=str, default='train_yolov8', help='Name for the training run directory')
    parser.add_argument('--save-period', type=int, default=10, help='Save checkpoint every X epochs')
    return parser.parse_args()


def train(args):
    """主训练函数"""
    # --- 1. 构造绝对路径 ---
    model_config_path = args.model_cfg
    dataset_yaml_path = os.path.join(ROOT_DIR, args.data_cfg)

    # --- 2. 加载模型 ---
    # 检查预训练模型是否存在
    if not os.path.exists(model_config_path):
        print(f"Error: Model config not found at {model_config_path}")
        raise FileNotFoundError(f"Model config not found at {model_config_path}")
    if not os.path.exists(dataset_yaml_path):
        print(f"Error: Dataset config not found at {dataset_yaml_path}")
        raise FileNotFoundError(f"Dataset config not found at {dataset_yaml_path}")
    model = YOLO(str(model_config_path))

    print(f"Starting training with model: {model_config_path}")
    print(f"Using dataset: {dataset_yaml_path}")

    # --- 3. 执行训练 ---
    filename = os.path.basename(model_config_path).split('.')[0]
    now_str = time.strftime("%Y%m%d_%H%M")
    results = model.train(
        data=dataset_yaml_path,
        workers=4,
        epochs=150,  # 增加训练周期到 150
        patience=30,  # 增加提前停止的耐心
        close_mosaic=10,  # 在最后10个epoch关闭mosaic
        save=True,
        save_period=10,
        name=f"train_{filename}_{now_str}_",
        exist_ok=True,  # 覆盖同名实验

        # --- 数据增强组合 ---
        # --- 1. 调整几何变换强度 ---
        degrees=15.0,  # 随机旋转 +/- 15 度 (更合理)
        translate=0.1,  # 随机平移 +/- 10% (更合理)
        scale=0.2,  # 随机缩放 +/- 20% (可以适当增大)
        shear=2.0,  # 随机错切 +/- 2 度 (保持较小)
        perspective=0.0,  # 对于工业平面场景，通常不需要透视变换
        flipud=0.0,  # 关闭垂直翻转，除非你的物体上下对称
        fliplr=0.5,  # 保留水平翻转
        # --- 2. 颜色空间变换 (保持或微调) ---
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        # --- 3. 开启高级增强---
        mosaic=1.0,  # 开启 Mosaic 数据增强 (将4张图拼接成一张)
        mixup=0.1,  # 以 10% 的概率开启 MixUp (将两张图混合)
        copy_paste=0.1  # 以 10% 的概率开启 Copy-Paste (复制物体实例)
    )

    print("\nTraining completed.")

    # --- 4. 验证并记录结果 ---
    print("Running validation on the best model...")
    # `model` 对象在 train 后会自动加载最佳权重，所以可以直接调用 val()
    val_metrics = model.val()

    # 构造结果字典
    summary = {
        "dataset": args.data_cfg,
        "model_config": args.model_cfg,
        "epochs_trained": args.epochs,
        "batch_size": args.batch_size,
        "results_directory": str(results.save_dir),
        "best_model_path": str(results.save_dir / 'weights' / 'best.pt'),
        "metrics": {
            "mAP50": float(f"{val_metrics.box.map50:.4f}"),
            "mAP50-95": float(f"{val_metrics.box.map:.4f}"),
            "precision": float(f"{val_metrics.box.p.mean():.4f}"),
            "recall": float(f"{val_metrics.box.r.mean():.4f}")
        }
    }

    # 将摘要保存为 JSON 文件
    summary_path = results.save_dir / 'training_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print("\n--- Validation Metrics ---")
    for key, value in summary['metrics'].items():
        print(f"{key}: {value}")
    print(f"--------------------------")
    print(f"Best model saved at: {summary['best_model_path']}")
    print(f"Training summary saved to: {summary_path}")


if __name__ == '__main__':
    # 示例运行命令
    # python train_yolo.py --data-cfg datasets/paper/dataset.yaml --epochs 100
    args = parse_args()
    train(args)
