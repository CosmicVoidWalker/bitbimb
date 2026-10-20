import cv2
import numpy as np

def calculate_histogram(image: np.ndarray) -> dict:
    """
    Calculates the 256-bin histogram for an image.
    Returns a dictionary of 1D arrays for each color channel.
    """
    hist_data = {}
    
    # Check if the image is RGB (3 channels)
    if len(image.shape) == 3:
        colors = ('r', 'g', 'b')
        for i, color in enumerate(colors):
            # calcHist arguments: images, channels, mask, histSize, ranges
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            hist_data[color] = hist.flatten()
    # Otherwise, it's Grayscale (2D array)
    else:
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist_data['gray'] = hist.flatten()
        
    return hist_data