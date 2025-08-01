# src/utils.py
import os
from loguru import logger
from tqdm import tqdm


def init_logging(log_path="logs/train.log"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger.add(log_path, rotation="500 KB")
    logger.info("Logger initialized.")


def get_image_paths(folder):
    """
    返回某个文件夹下所有图像文件的路径
    """
    valid_exts = [".jpg", ".jpeg", ".png"]
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in valid_exts
    ]