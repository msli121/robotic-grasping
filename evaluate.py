import argparse
import datetime
import logging
import os
import time

import numpy as np
import torch.utils.data

from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data import get_dataset
from utils.dataset_processing import evaluation, grasp
from utils.visualisation.plot import save_results

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate networks')

    # Network
    parser.add_argument('--network', metavar='N', type=str, nargs='+',
                        help='Path to saved networks to evaluate')
    parser.add_argument('--input-size', type=int, default=224,
                        help='Input image size for the network')

    # Dataset
    parser.add_argument('--dataset', type=str,
                        help='Dataset Name ("cornell" or "jaquard")')
    parser.add_argument('--dataset-path', type=str,
                        help='Path to dataset')
    parser.add_argument('--use-depth', type=int, default=1,
                        help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1,
                        help='Use RGB image for evaluation (1/0)')
    parser.add_argument('--augment', action='store_true',
                        help='Whether data augmentation should be applied')
    parser.add_argument('--split', type=float, default=0.9,
                        help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-shuffle', action='store_true', default=False,
                        help='Shuffle the dataset')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Dataset workers')

    # Evaluation
    parser.add_argument('--n-grasps', type=int, default=1,
                        help='Number of grasps to consider per image')
    parser.add_argument('--iou-threshold', type=float, default=0.25,
                        help='Threshold for IOU matching')
    parser.add_argument('--iou-eval', action='store_true',
                        help='Compute success based on IoU metric.')
    parser.add_argument('--jacquard-output', action='store_true',
                        help='Jacquard-dataset style output')

    # Misc.
    parser.add_argument('--vis', action='store_true',
                        help='Visualise the network output')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')
    parser.add_argument('--random-seed', type=int, default=123,
                        help='Random seed for numpy')

    # log
    parser.add_argument('--logdir', type=str, default='logs/',
                        help='Log directory')

    args = parser.parse_args()

    if args.jacquard_output and args.dataset != 'jacquard':
        raise ValueError('--jacquard-output can only be used with the --dataset jacquard option.')
    if args.jacquard_output and args.augment:
        raise ValueError('--jacquard-output can not be used with data augmentation.')

    return args


def run():
    # Parse args
    args = parse_args()

    # Init log
    init_log(args)

    # Print Args
    logging.info(f'Arguments: {args}')

    # Get the compute device
    device = get_device(args.force_cpu)

    # Load Dataset
    logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    Dataset = get_dataset(args.dataset)
    test_dataset = Dataset(args.dataset_path,
                           output_size=args.input_size,
                           ds_rotate=args.ds_rotate,
                           random_rotate=args.augment,
                           random_zoom=args.augment,
                           include_depth=args.use_depth,
                           include_rgb=args.use_rgb)

    indices = list(range(test_dataset.length))
    split = int(np.floor(args.split * test_dataset.length))
    if args.ds_shuffle:
        np.random.seed(args.random_seed)
        np.random.shuffle(indices)
    val_indices = indices[split:]
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)
    logging.info('Validation size: {}'.format(len(val_indices)))

    test_data = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        sampler=val_sampler
    )
    logging.info('Load test data done!')

    last_print_time = time.time()
    total = len(test_data)

    for network in args.network:
        logging.info('\nEvaluating model {}'.format(network))

        # Load Network
        net = torch.load(network)

        results = {'correct': 0, 'failed': 0}

        if args.jacquard_output:
            jo_fn = network + '_jacquard_output.txt'
            with open(jo_fn, 'w') as f:
                pass

        start_time = time.time()

        with torch.no_grad():
            for idx, (x, y, didx, rot, zoom) in enumerate(test_data):
                print("x ", x.shape, x.min(), x.max())
                xc = x.to(device)
                yc = [yi.to(device) for yi in y]
                lossd = net.compute_loss(xc, yc)

                q_img, ang_img, width_img = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                                lossd['pred']['sin'], lossd['pred']['width'])

                if args.iou_eval:
                    s = evaluation.calculate_iou_match(q_img, ang_img, test_data.dataset.get_gtbb(didx, rot, zoom),
                                                       no_grasps=args.n_grasps,
                                                       grasp_width=width_img,
                                                       threshold=args.iou_threshold
                                                       )
                    if s:
                        results['correct'] += 1
                    else:
                        results['failed'] += 1

                    # 每隔10秒计算进度信息打印
                    now_time = time.time()
                    if now_time - last_print_time > 10:
                        progress = ((idx + 1) / total) * 100
                        # current_iou = correct / (idx + 1) if idx > 0 else 0
                        current_iou = ' %d/%d = %f' % (results['correct'],
                                                       results['correct'] + results['failed'],
                                                       results['correct'] / (
                                                               results['correct'] + results['failed']))
                        logging.info('Progress: {:.2f}% - Current IOU: {}'.format(progress, current_iou))
                        last_print_time = now_time

                if args.jacquard_output:
                    grasps = grasp.detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=1)
                    with open(jo_fn, 'a') as f:
                        for g in grasps:
                            f.write(test_data.dataset.get_jname(didx) + '\n')
                            f.write(g.to_jacquard(scale=1024 / 300) + '\n')

                if args.vis:
                    save_results(
                        rgb_img=test_data.dataset.get_rgb(didx, rot, zoom, normalise=False),
                        depth_img=test_data.dataset.get_depth(didx, rot, zoom),
                        grasp_q_img=q_img,
                        grasp_angle_img=ang_img,
                        no_grasps=args.n_grasps,
                        grasp_width_img=width_img
                    )

        avg_time = (time.time() - start_time) / len(test_data)
        logging.info('Average evaluation time per image: {}ms'.format(avg_time * 1000))

        if args.iou_eval:
            logging.info('IOU Results: %d/%d = %f' % (results['correct'],
                                                      results['correct'] + results['failed'],
                                                      results['correct'] / (results['correct'] + results['failed'])))

        if args.jacquard_output:
            logging.info('Jacquard output saved to {}'.format(jo_fn))

        del net
        torch.cuda.empty_cache()


def init_log(args):
    # Set-up output directories
    dt = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    net_desc = '{}_{}'.format(dt, args.dataset)
    save_folder = os.path.join(args.logdir, 'evaluate', net_desc)
    os.makedirs(save_folder, exist_ok=True)
    # Initialize logging
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        filename="{0}/{1}.log".format(save_folder, 'log'),
        format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('[%(asctime)s] %(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)


if __name__ == '__main__':
    # baseline cornell
    # python evaluate.py --network trained-pretrained_models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98 --dataset cornell --dataset-path D:\\datasets\\cornell_grasp --iou-eval

    # python evaluate.py --network trained-pretrained_models/jacquard-rgbd-grconvnet3-drop0-ch32/epoch_48_iou_0.93 --dataset cornell --dataset-path D:\\datasets\\cornell_grasp --iou-eval
    # python evaluate.py --network logs/250814_1440_training_cornell_grconvnet_goa/best_iou_epoch_21_iou_0.9153 --dataset cornell --dataset-path D:\\datasets\\cornell_grasp --iou-eval
    # python evaluate.py --network logs/250814_1440_training_cornell_grconvnet_goa/best_iou_epoch_21_iou_0.9153 --dataset cornell --dataset-path D:\\datasets\\cornell_grasp --iou-eval
    # python evaluate.py --network logs/250811_0138_training_cornell_grconvnet3/epoch_33_iou_0.9209 --dataset cornell --dataset-path D:\\datasets\\cornell_grasp --iou-eval
    # python evaluate.py --network D:\PycharmProjects\robotic-grasping\trained-pretrained_models\jacquard-rgbd-grconvnet3-drop0-ch32\epoch_35_iou_0.92 --dataset jacquard --dataset-path D:\\datasets\\Jacquard --iou-eval --input-size 300 --vis
    run()
