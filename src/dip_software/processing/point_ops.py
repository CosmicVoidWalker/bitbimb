import cv2
import numpy as np

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Converts image to 2D grayscale if it has 3 channels."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image.copy()

def apply_negative(image: np.ndarray) -> np.ndarray:
    """Inverts pixel intensities."""
    return 255 - image

def log_transformation(image: np.ndarray) -> np.ndarray:
    """Expands dark pixels and compresses bright pixels (s = c * log(1 + r))."""
    c = 255 / np.log(1 + np.max(image))
    log_image = c * (np.log(1 + image.astype(np.float32)))
    return np.array(log_image, dtype=np.uint8)

def gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
    """Applies power-law (gamma) correction using a Look-Up Table."""
    gamma = max(0.01, gamma) # Prevent division by zero
    inv_gamma = 1.0 / gamma
    
    # Precompute mapping for 0-255 inputs
    table = np.array([((i / 255.0) ** inv_gamma) * 255 
                      for i in np.arange(0, 256)]).astype("uint8")
    
    return cv2.LUT(image, table)