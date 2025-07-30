# face_detector.py

import os
import cv2
import torch
import numpy as np
from PIL import Image, UnidentifiedImageError
from facenet_pytorch import MTCNN
from torchvision import transforms
from typing import Dict, Tuple

class FaceRegionExtractor:
    def __init__(self, cache_dir="/root/Project/RCNN Models/cache/regions", image_size=160, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mtcnn = MTCNN(keep_all=False, device=self.device)  # 只提取主脸
        self.cache_dir = cache_dir
        self.image_size = image_size

        os.makedirs(self.cache_dir, exist_ok=True)

    def extract(self, img_path: str) -> Dict[str, Image.Image]:
        """
        提取图像中的主脸区域以及五官区域（眼、嘴）。
        返回一个字典：{"full": Image, "eyes": Image, "mouth": Image}
        """

        # 缓存路径
        cache_base = os.path.join(self.cache_dir, os.path.splitext(os.path.basename(img_path))[0])
        region_paths = {
            "full": cache_base + "_face.jpg",
            "eyes": cache_base + "_eyes.jpg",
            "mouth": cache_base + "_mouth.jpg"
        }

        # 如果缓存存在
        if all(os.path.exists(p) for p in region_paths.values()):
            return {k: Image.open(p).convert("RGB") for k, p in region_paths.items()}

        # 检测主脸
        try:
            img = Image.open(img_path).convert("RGB")
        except (OSError, UnidentifiedImageError) as e:
            raise ValueError(f"Failed to open image: {img_path}") from e

        result = self.mtcnn.detect(img, landmarks=True)
        if not isinstance(result, tuple) or len(result) != 3:
            raise ValueError(f"MTCNN did not return expected outputs for image: {img_path}")

        boxes, probs, landmarks = result

        if boxes is None or len(boxes) == 0 or landmarks is None or landmarks[0] is None:
            raise ValueError(f"No face detected in image: {img_path}")

        # 主脸区域裁剪
        box = boxes[0]  # 只用最大脸
        x1, y1, x2, y2 = map(int, box)
        face = img.crop((x1, y1, x2, y2)).resize((self.image_size, self.image_size))

        # 面部关键点
        lm = landmarks[0]  # (5, 2)
        left_eye, right_eye, nose, left_mouth, right_mouth = lm

        # 眼睛区域
        eye_x1 = int(min(left_eye[0], right_eye[0]) - 10)
        eye_y1 = int(min(left_eye[1], right_eye[1]) - 10)
        eye_x2 = int(max(left_eye[0], right_eye[0]) + 10)
        eye_y2 = int(max(left_eye[1], right_eye[1]) + 10)

        eyes = img.crop((eye_x1, eye_y1, eye_x2, eye_y2)).resize((self.image_size, self.image_size))

        # 嘴巴区域
        mouth_x1 = int(min(left_mouth[0], right_mouth[0]) - 10)
        mouth_y1 = int(min(left_mouth[1], right_mouth[1]) - 10)
        mouth_x2 = int(max(left_mouth[0], right_mouth[0]) + 10)
        mouth_y2 = int(max(left_mouth[1], right_mouth[1]) + 10)

        mouth = img.crop((mouth_x1, mouth_y1, mouth_x2, mouth_y2)).resize((self.image_size, self.image_size))

        # 保存缓存
        # face.save(region_paths["full"])
        # eyes.save(region_paths["eyes"])
        # mouth.save(region_paths["mouth"])

        return {
            "full": face,
            "eyes": eyes,
            "mouth": mouth
        }