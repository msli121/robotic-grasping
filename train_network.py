# -*- coding: utf-8 -*-
import argparse
import datetime
import json
import logging
import os
import sys
import random
import math

import cv2
import numpy as np
import tensorboardX
import torch
import torch.optim as optim
import torch.utils.data
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Subset, RandomSampler, SequentialSampler, DataLoader
from tqdm import tqdm

from hardware.device import get_device
from inference.models import get_network
from inference.post_process import post_process_output
from utils.data import get_dataset
from utils.dataset_processing import evaluation
from utils.visualisation.gridshow import gridshow


# ------------------------ Args ------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Train network')

    # Network
    parser.add_argument('--network', type=str, default='grconvnet3',
                        help='Network name in inference/pretrained_models')
    parser.add_argument('--input-size', type=int, default=224,
                        help='Input image size for the network')
    parser.add_argument('--use-depth', type=int, default=1,
                        help='Use Depth image for training (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1,
                        help='Use RGB image for training (1/0)')
    parser.add_argument('--use-dropout', type=int, default=1,
                        help='Use dropout for training (1/0)')
    parser.add_argument('--dropout-prob', type=float, default=0.1,
                        help='Dropout prob for training (0-1)')
    parser.add_argument('--channel-size', type=int, default=32,
                        help='Internal channel size for the network')
    parser.add_argument('--iou-threshold', type=float, default=0.25,
                        help='Threshold for IOU matching')

    # Datasets
    parser.add_argument('--dataset', type=str,
                        help='Dataset Name ("cornell" or "jaquard")')
    parser.add_argument('--dataset-path', type=str,
                        help='Path to dataset')
    parser.add_argument('--split', type=float, default=0.9,
                        help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-shuffle', action='store_true', default=False,
                        help='Shuffle the dataset')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Dataset workers')

    # Training
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs')
    parser.add_argument('--batches-per-epoch', type=int, default=1000,
                        help='Batches per Epoch')
    parser.add_argument('--optim', type=str, default='adamw',
                        help='Optmizer for the training. (adamw or adam or SGD)')

    # Logging etc.
    parser.add_argument('--description', type=str, default='',
                        help='Training description')
    parser.add_argument('--logdir', type=str, default='logs/',
                        help='Log directory')
    parser.add_argument('--vis', action='store_true',
                        help='Visualise the training process')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')
    parser.add_argument('--random-seed', type=int, default=123,
                        help='Random seed for numpy')

    # 改进网络开关（保持兼容）
    parser.add_argument('--upconv', type=int, default=0)
    parser.add_argument('--unet', type=int, default=0)
    parser.add_argument('--fpn', type=int, default=0)
    parser.add_argument('--goa', type=int, default=0)
    parser.add_argument('--cbam', type=int, default=0)
    parser.add_argument('--aff', type=int, default=0)
    parser.add_argument('--spdconv', type=int, default=0)
    parser.add_argument('--spd-scale', type=int, default=2)

    # 新增训练策略参数（保持你原有默认）
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay for AdamW optimizer')
    parser.add_argument('--lr-patience', type=int, default=5,
                        help='Patience for ReduceLROnPlateau (epochs)')
    parser.add_argument('--early-stop-patience', type=int, default=30,
                        help='Patience for early stopping (epochs)')

    return parser.parse_args()


# ------------------------ Utils ------------------------
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    """
    让每个 dataloader worker 拥有不同的随机种子，避免小数据集上增强完全同步。
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def save_model_arch(net, save_folder):
    try:
        arch_path = os.path.join(save_folder, 'arch.txt')
        with open(arch_path, 'w') as f:
            f.write(str(net))
            f.write('\n\n')
            total_params = sum(p.numel() for p in net.parameters())
            trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
            f.write(f"Total Parameters: {total_params:,}\n")
            f.write(f"Trainable Parameters: {trainable_params:,}\n")
        logging.info(f"Model architecture saved to {arch_path}")
        logging.info(f"Total Parameters: {total_params:,}")
        logging.info(f"Trainable Parameters: {trainable_params:,}")
    except Exception as e:
        logging.warning(f"Failed to save model architecture: {e}")


# ------------------------ Data Loaders ------------------------
def make_dataloaders(args, save_folder):
    """
    为 train/val 分别实例化数据集；小数据集使用带 replacement 的 RandomSampler；
    固化索引划分到磁盘，保证复现实验。
    """
    Dataset = get_dataset(args.dataset)

    # 训练集（开启随机增强）
    train_dataset = Dataset(
        args.dataset_path,
        output_size=args.input_size,
        ds_rotate=args.ds_rotate,
        random_rotate=True,
        random_zoom=True,
        include_depth=args.use_depth,
        include_rgb=args.use_rgb
    )
    # 验证集（关闭随机增强）
    val_dataset = Dataset(
        args.dataset_path,
        output_size=args.input_size,
        ds_rotate=args.ds_rotate,
        random_rotate=False,
        random_zoom=False,
        include_depth=args.use_depth,
        include_rgb=args.use_rgb
    )

    # 固化划分：读写 split_indices.json
    split_file = os.path.join(save_folder, 'split_indices.json')
    if os.path.exists(split_file):
        with open(split_file, 'r') as f:
            split_indices = json.load(f)
        train_indices = split_indices['train']
        val_indices = split_indices['val']
        logging.info(f'Loaded existing split indices from {split_file}')
    else:
        total_len = train_dataset.length if hasattr(train_dataset, 'length') else len(train_dataset)
        indices = list(range(total_len))
        if args.ds_shuffle:
            np.random.seed(args.random_seed)
            np.random.shuffle(indices)
        split = int(np.floor(args.split * total_len))
        train_indices, val_indices = indices[:split], indices[split:]
        with open(split_file, 'w') as f:
            json.dump({'train': train_indices, 'val': val_indices}, f, indent=2)
        logging.info(f'Saved split indices to {split_file}')

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    # 训练采样器：replacement 精准控制 epoch 大小
    samples_per_epoch = args.batches_per_epoch * args.batch_size
    train_sampler = RandomSampler(train_subset, replacement=True, num_samples=samples_per_epoch)

    # 验证采样器：顺序确保稳定
    val_sampler = SequentialSampler(val_subset)

    pin_memory = True
    persistent_workers = (args.num_workers > 0)
    generator = torch.Generator()
    generator.manual_seed(args.random_seed)

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=2 if args.num_workers > 0 else None,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=generator
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=1,
        sampler=val_sampler,
        num_workers=max(0, min(2, args.num_workers)),
        pin_memory=pin_memory,
        persistent_workers=False,
        drop_last=False
    )

    total_len = train_dataset.length if hasattr(train_dataset, 'length') else len(train_dataset)
    logging.info(f"Dataset size (total): {total_len}")
    logging.info(f"Train/Val split: {len(train_indices)}/{len(val_indices)}")
    logging.info(
        f"Train samples per epoch (effective): {samples_per_epoch} = {args.batches_per_epoch} x {args.batch_size}")

    return train_loader, val_loader


# ------------------------ Validation ------------------------
@torch.no_grad()
def validate(net, device, val_data, iou_threshold):
    """
    Run validation.
    """
    net.eval()
    results = {'correct': 0, 'failed': 0, 'loss': 0, 'losses': {}}
    ld = len(val_data)

    # --- 解开 Subset，拿到底层真正的数据集 ---
    base_ds = val_data.dataset
    while isinstance(base_ds, Subset):
        base_ds = base_ds.dataset

    for x, y, didx, rot, zoom_factor in val_data:
        xc = x.to(device, non_blocking=True)
        yc = [yy.to(device, non_blocking=True) for yy in y]
        lossd = net.compute_loss(xc, yc)
        loss = lossd['loss']

        results['loss'] += loss.item() / ld
        for ln, l in lossd['losses'].items():
            results['losses'].setdefault(ln, 0.0)
            results['losses'][ln] += float(l.item()) / ld

        q_out, ang_out, w_out = post_process_output(
            lossd['pred']['pos'], lossd['pred']['cos'],
            lossd['pred']['sin'], lossd['pred']['width']
        )

        # 用底层数据集的 get_gtbb（注意：didx 就是底层数据集的样本索引）
        gt = base_ds.get_gtbb(didx, rot, zoom_factor)

        s = evaluation.calculate_iou_match(q_out, ang_out, gt,
                                           no_grasps=1, grasp_width=w_out, threshold=iou_threshold
                                           )

        if s:
            results['correct'] += 1
        else:
            results['failed'] += 1

    return results


# ------------------------ Training ------------------------
def train(epoch, net, device, train_data, optimizer, batches_per_epoch, vis=False):
    """
    Run one training epoch
    """
    results = {'loss': 0.0, 'losses': {}}
    net.train()

    scaler = GradScaler(enabled=(device.type == 'cuda'))
    steps = 0

    with tqdm(total=batches_per_epoch, desc=f"Epoch {epoch + 1:02d}", leave=True) as pbar:
        for x, y, _, _, _ in train_data:
            if steps >= batches_per_epoch:
                break
            steps += 1

            xc = x.to(device, non_blocking=True)
            yc = [yy.to(device, non_blocking=True) for yy in y]

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == 'cuda')):
                lossd = net.compute_loss(xc, yc)
                loss = lossd['loss']

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            results['loss'] += loss.item()
            for ln, l in lossd['losses'].items():
                results['losses'].setdefault(ln, 0.0)
                results['losses'][ln] += float(l.item())

            pbar.set_postfix(loss=f"{loss.item():.4f}")
            pbar.update(1)

            if vis:
                imgs = []
                n_img = min(4, x.shape[0])
                for idx in range(n_img):
                    imgs.extend([x[idx,].numpy().squeeze()] + [yi[idx,].numpy().squeeze() for yi in y] + [
                        x[idx,].numpy().squeeze()] + [pc[idx,].detach().cpu().numpy().squeeze() for pc in
                                                      lossd['pred'].values()])
                gridshow('Display', imgs,
                         [(xc.min().item(), xc.max().item()), (0.0, 1.0), (0.0, 1.0), (-1.0, 1.0),
                          (0.0, 1.0)] * 2 * n_img,
                         [cv2.COLORMAP_BONE] * 10 * n_img, 10)
                cv2.waitKey(2)

    results['loss'] /= max(1, steps)
    for l in results['losses']:
        results['losses'][l] /= max(1, steps)

    return results


# ------------------------ Main ------------------------
def run():
    args = parse_args()

    # Device
    device = get_device(args.force_cpu)

    # Network
    logging.info(f'Loading Network... network = {args.network}')
    input_channels = 1 * args.use_depth + 3 * args.use_rgb
    network = get_network(args.network)
    if args.network.lower() in ['grconvnet_goa']:
        net = network(
            input_channels=input_channels,
            dropout=bool(args.use_dropout),
            prob=args.dropout_prob,
            channel_size=args.channel_size,
            use_upconv=bool(args.upconv),
            use_unet=bool(args.unet),
            use_fpn=bool(args.fpn),
            use_cbam=bool(args.cbam),
            use_goa=bool(args.goa),
            use_aff=bool(args.aff),
            use_spd=bool(args.spdconv),
            spd_scale=args.spd_scale,
        )
    else:
        net = network(
            input_channels=input_channels,
            dropout=args.use_dropout,
            prob=args.dropout_prob,
            channel_size=args.channel_size
        )
    net = net.to(device)
    logging.info('Done')

    # Output dirs & TB
    dt = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    net_config_name = net.get_config_name()
    net_desc = f"{dt}_{'_'.join(args.description.split())}_{args.input_size}_{net_config_name}"
    save_folder = os.path.join(args.logdir, net_desc)
    os.makedirs(save_folder, exist_ok=True)
    tb = tensorboardX.SummaryWriter(save_folder)

    # Save args
    params_path = os.path.join(save_folder, 'commandline_args.json')
    with open(params_path, 'w') as f:
        json.dump(vars(args), f)

    # Logging
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join(save_folder, 'log.log'),
        format='[%(asctime)s] %(levelname)s - %(message)s - {%(pathname)s:%(lineno)d}',
        datefmt='%H:%M:%S'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s'))
    logging.getLogger('').addHandler(console)

    # Reproducibility & CUDNN
    set_global_seed(args.random_seed)
    torch.backends.cudnn.benchmark = True  # 若追求完全可复现可设为 False 并置 deterministic=True
    # torch.backends.cudnn.deterministic = True

    # Data
    logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    train_data, val_data = make_dataloaders(args, save_folder)
    logging.info('Done')

    # Optimizer
    if args.optim.lower() == 'adamw':
        logging.info(f"Using AdamW Optimizer with lr={args.lr} and weight_decay={args.weight_decay}")
        optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim.lower() == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    elif args.optim.lower() == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    else:
        raise NotImplementedError('Optimizer {} is not implemented'.format(args.optim))

    # LR Scheduler
    logging.info(f"Using ReduceLROnPlateau scheduler with patience={args.lr_patience}")
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=args.lr_patience)

    # Save model arch
    save_model_arch(net, save_folder)

    # Early stopping
    best_iou = 0.0
    patience_counter = 0

    for epoch in range(args.epochs):
        # Train
        train_results = train(epoch, net, device, train_data, optimizer, args.batches_per_epoch, vis=args.vis)

        # TB: train
        tb.add_scalar('loss/train_loss', train_results['loss'], epoch)
        for n, l in train_results['losses'].items():
            tb.add_scalar('train_loss/' + n, l, epoch)
        tb.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # Validate
        logging.info(f"Epoch {epoch + 1}/{args.epochs} - Validating...")
        test_results = validate(net, device, val_data, args.iou_threshold)
        iou = test_results['correct'] / max(1, (test_results['correct'] + test_results['failed']))
        logging.info(
            f"Epoch {epoch + 1}/{args.epochs} - Validation Result: {test_results['correct']}/{test_results['correct'] + test_results['failed']} = {iou:.4f}"
        )

        # TB: val
        tb.add_scalar('loss/IOU', iou, epoch)
        tb.add_scalar('loss/val_loss', test_results['loss'], epoch)
        for n, l in test_results['losses'].items():
            tb.add_scalar('val_loss/' + n, l, epoch)

        # Scheduler on IoU
        scheduler.step(iou)

        # Save best
        if iou > best_iou:
            logging.info(f" >> IOU improved from {best_iou:.4f} to {iou:.4f}. Saving best model...")
            best_iou = iou
            # 删除旧的 best
            for f in os.listdir(save_folder):
                if f.startswith('best_model'):
                    try:
                        os.remove(os.path.join(save_folder, f))
                    except Exception:
                        pass
            torch.save(net, os.path.join(save_folder, f'best_model_epoch_{epoch + 1:02d}_iou_{iou:.4f}.pth'))
            patience_counter = 0
        else:
            patience_counter += 1

        # Periodic checkpoint
        if epoch % 5 == 0:
            logging.info(f"Epoch {epoch + 1}/{args.epochs} - Checkpoint saved")
            torch.save(net, os.path.join(save_folder, f'checkpoint_epoch_{epoch + 1:02d}_iou_{iou:.4f}.pth'))

        # Early stop
        if patience_counter >= args.early_stop_patience:
            logging.info(
                f"Epoch {epoch + 1}/{args.epochs} - Early stopping triggered after {patience_counter} epochs with no improvement."
            )
            break


if __name__ == '__main__':
    run()
