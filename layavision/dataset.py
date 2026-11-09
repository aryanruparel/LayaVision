"""
LayaVision Dataset Utilities
============================
Helpers for loading image datasets, creating synthetic test samples,
and preparing batches for training and evaluation.
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset


def generate_synthetic_image(category: str, size: Tuple[int, int] = (224, 224)) -> Image.Image:
    """
    Generate synthetic benchmark images with distinct visual patterns
    for quick verification, testing, and continuous integration.
    """
    img = Image.new("RGB", size, color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    w, h = size

    cat_lower = category.lower()
    if "car" in cat_lower or "vehicle" in cat_lower:
        # Draw a car-like shape (blue body, yellow wheels)
        draw.rectangle([w * 0.2, h * 0.5, w * 0.8, h * 0.75], fill=(30, 100, 220))
        draw.polygon([(w * 0.35, h * 0.5), (w * 0.45, h * 0.3), (w * 0.65, h * 0.3), (w * 0.75, h * 0.5)], fill=(120, 180, 255))
        draw.ellipse([w * 0.25, h * 0.7, w * 0.4, h * 0.85], fill=(40, 40, 40))
        draw.ellipse([w * 0.6, h * 0.7, w * 0.75, h * 0.85], fill=(40, 40, 40))
    elif "dog" in cat_lower or "puppy" in cat_lower:
        # Draw a dog-like shape (golden-brown body and floppy ears)
        draw.ellipse([w * 0.3, h * 0.4, w * 0.7, h * 0.8], fill=(185, 122, 60))
        draw.ellipse([w * 0.45, h * 0.25, w * 0.75, h * 0.55], fill=(205, 142, 80))
        # floppy ear
        draw.polygon([(w * 0.65, h * 0.3), (w * 0.78, h * 0.5), (w * 0.7, h * 0.55)], fill=(150, 90, 40))
        # eye & nose
        draw.ellipse([w * 0.55, h * 0.35, w * 0.6, h * 0.4], fill=(20, 20, 20))
        draw.ellipse([w * 0.65, h * 0.42, w * 0.72, h * 0.48], fill=(20, 20, 20))
    elif "cat" in cat_lower or "kitten" in cat_lower:
        # Draw a cat-like shape (orange tabby with pointy ears)
        draw.ellipse([w * 0.3, h * 0.45, w * 0.7, h * 0.82], fill=(230, 140, 50))
        draw.ellipse([w * 0.35, h * 0.25, w * 0.65, h * 0.52], fill=(240, 150, 60))
        # pointy ears
        draw.polygon([(w * 0.38, h * 0.28), (w * 0.34, h * 0.15), (w * 0.48, h * 0.26)], fill=(200, 100, 30))
        draw.polygon([(w * 0.52, h * 0.26), (w * 0.66, h * 0.15), (w * 0.62, h * 0.28)], fill=(200, 100, 30))
        # whiskers & eyes
        draw.ellipse([w * 0.42, h * 0.35, w * 0.46, h * 0.4], fill=(30, 200, 30))
        draw.ellipse([w * 0.54, h * 0.35, w * 0.58, h * 0.4], fill=(30, 200, 30))
    elif "apple" in cat_lower or "fruit" in cat_lower:
        # Red apple with green leaf
        draw.ellipse([w * 0.25, h * 0.25, w * 0.75, h * 0.8], fill=(220, 30, 30))
        draw.line([w * 0.5, h * 0.25, w * 0.5, h * 0.15], fill=(90, 50, 20), width=5)
        draw.ellipse([w * 0.5, h * 0.12, w * 0.65, h * 0.22], fill=(40, 180, 40))
    else:
        # Generic colored shape with label
        draw.rectangle([w * 0.2, h * 0.2, w * 0.8, h * 0.8], fill=(100, 140, 200))
        draw.text((w * 0.3, h * 0.45), category[:10], fill=(255, 255, 255))

    return img


class ImageClassificationDataset(Dataset):
    """
    Standard directory-based image dataset for object classification.
    Expects structure:
      root_dir/
        class_a/
          img1.jpg
          img2.png
        class_b/
          img3.jpg
    """

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        self.classes: List[str] = []

        if os.path.exists(root_dir):
            self.classes = sorted([
                d for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            ])
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

            valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            for c in self.classes:
                cdir = os.path.join(root_dir, c)
                for f in os.listdir(cdir):
                    ext = os.path.splitext(f)[1].lower()
                    if ext in valid_exts:
                        self.samples.append((os.path.join(cdir, f), self.class_to_idx[c]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
