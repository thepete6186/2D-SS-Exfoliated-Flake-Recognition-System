#!/usr/bin/env python3
"""
load.py

Load and preprocess TMD sample images.
"""

import numpy as np
import cv2
from pathlib import Path


def load_image(path):
    """Load an image from disk and return it in RGB (HxWx3, uint8)."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def find_sample_images(input_dir):
    """Find all tmd_sample images in the given directory."""
    input_dir = Path(input_dir)
    image_paths = sorted(input_dir.glob("tmd_sample_*.jpg"))
    return image_paths