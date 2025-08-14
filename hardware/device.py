import logging

import torch

logging.basicConfig(level=logging.INFO)


def get_device(force_cpu):
    # Check if CUDA can be used
    if torch.cuda.is_available() and not force_cpu:
        logging.info("CUDA detected. Running with GPU acceleration.")
        device = torch.device("cuda")
        # 打印GPU信息 包括型号、内存、计算能力
        logging.info("GPU Info: {}".format(torch.cuda.get_device_name(0)))
        logging.info("GPU Memory: {}MB".format(torch.cuda.get_device_properties(0).total_memory / 1024 ** 2))
    elif force_cpu:
        logging.info("CUDA detected, but overriding with option '--cpu'. Running with only CPU.")
        device = torch.device("cpu")
    else:
        logging.info("CUDA is *NOT* detected. Running with only CPU.")
        device = torch.device("cpu")
    return device
