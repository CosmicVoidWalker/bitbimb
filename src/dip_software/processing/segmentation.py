import cv2
import numpy as np

def global_threshold(image: np.ndarray, threshold_value: int = 127) -> np.ndarray:
    """Applies a fixed-level threshold to separate foreground from background."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
        
    # Pixels above threshold_value become 255 (white), others become 0 (black)
    _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    return thresh

def otsu_threshold(image: np.ndarray) -> np.ndarray:
    """Automatically calculates the optimal threshold value using Otsu's method."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
        
    # Otsu's algorithm analyzes the histogram bimodal distribution to find the best cut
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh