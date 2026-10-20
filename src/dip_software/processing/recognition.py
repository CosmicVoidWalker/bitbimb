import cv2
import numpy as np

def extract_objects(image: np.ndarray, min_area: int = 100) -> np.ndarray:
    """Finds contours in a binary image and draws bounding boxes around objects."""
    # Ensure we have a strictly binary image for contour detection
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        output = image.copy() # Keep color to draw green boxes on original
    else:
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        output = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) # Convert 2D array to RGB

    # Extract object boundaries
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by area and draw bounding boxes
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
    return output