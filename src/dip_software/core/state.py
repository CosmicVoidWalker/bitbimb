import cv2
import numpy as np

class ImageState:
    def __init__(self):
        self.current_image = None
        self.history = []

    def load(self, filepath: str) -> bool:
        """Loads an image safely and resets history."""
        img = cv2.imread(filepath)
        if img is None:
            return False
        
        self.current_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.history.clear()
        return True

    def update(self, new_image: np.ndarray):
        """Pushes the current image to history before applying the new one."""
        if self.current_image is not None:
            self.history.append(self.current_image.copy())
        self.current_image = new_image

    def undo(self) -> bool:
        """Pops the last image state from the stack."""
        if self.history:
            self.current_image = self.history.pop()
            return True
        return False