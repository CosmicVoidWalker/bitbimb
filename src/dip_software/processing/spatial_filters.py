import cv2
import numpy as np

def mean_blur(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Applies a basic mean (average) blur using a ksize x ksize kernel."""
    # Ensure kernel size is always an odd number
    if ksize % 2 == 0: 
        ksize += 1
    return cv2.blur(image, (ksize, ksize))

def gaussian_blur(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Applies a Gaussian blur, reducing high-frequency noise."""
    if ksize % 2 == 0: 
        ksize += 1
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def sobel_edge(image: np.ndarray) -> np.ndarray:
    """Detects edges by calculating spatial gradients (X and Y directions)."""
    # Edge detection is typically performed on grayscale images
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
        
    # Calculate gradients in X and Y directions
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Convert back to absolute 8-bit values and combine them
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    
    # Combine the two gradients equally
    return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

def laplacian_edge(image: np.ndarray) -> np.ndarray:
    """Detects edges using the 2nd derivative (Laplacian)."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
        
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return cv2.convertScaleAbs(laplacian)