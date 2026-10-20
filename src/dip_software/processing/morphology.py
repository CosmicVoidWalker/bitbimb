import cv2
import numpy as np

def _get_kernel(ksize: int) -> np.ndarray:
    """Helper function to generate a square structuring element."""
    if ksize % 2 == 0: 
        ksize += 1
    return np.ones((ksize, ksize), np.uint8)

def erode(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Erodes boundaries of foreground objects (shrinks white areas)."""
    return cv2.erode(image, _get_kernel(ksize), iterations=1)

def dilate(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Expands boundaries of foreground objects (grows white areas)."""
    return cv2.dilate(image, _get_kernel(ksize), iterations=1)

def opening(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Erosion followed by Dilation. Great for removing external salt noise."""
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, _get_kernel(ksize))

def closing(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Dilation followed by Erosion. Great for filling small internal holes (pepper noise)."""
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, _get_kernel(ksize))