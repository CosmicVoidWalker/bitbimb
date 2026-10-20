import cv2
import numpy as np
import os

def save_image(image: np.ndarray, filepath: str, quality: int = 90) -> bool:
    """
    Saves the image to disk. 
    Applies lossy compression if the format is JPEG.
    """
    if image is None:
        return False
        
    # Convert RGB back to BGR for OpenCV's file writer
    if len(image.shape) == 3:
        out_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        out_img = image
        
    # Check file extension
    _, ext = os.path.splitext(filepath)
    ext = ext.lower()
    
    try:
        if ext in ['.jpg', '.jpeg']:
            # Apply JPEG compression quality (0-100, lower is smaller file/lower quality)
            return cv2.imwrite(filepath, out_img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        elif ext == '.png':
            # PNG uses lossless compression (0-9, higher is smaller file but slower)
            return cv2.imwrite(filepath, out_img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
        else:
            # Default save for BMP, TIFF, etc.
            return cv2.imwrite(filepath, out_img)
    except Exception as e:
        print(f"Failed to save image: {e}")
        return False