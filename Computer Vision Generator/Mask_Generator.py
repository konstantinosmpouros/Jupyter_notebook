import cv2
import numpy as np
from PIL import Image
import os

os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'


class MaskGeneration:

    def __init__(self, image):
        self.image = image

        # Convert the PIL image to a NumPy array compatible with OpenCV and from RGB to BGR
        self.image_cv = np.array(image)
        self.image_cv = cv2.cvtColor(self.image_cv, cv2.COLOR_RGB2BGR)

        # Initialize the mask with the same size as the image (height, width)
        self.mask = np.zeros(self.image_cv.shape[:2], dtype=np.uint8)

        # List to store the history of mask states for undo functionality
        self.history = []

        # Global variables to store the drawing state
        self.drawing = False  # True if the mouse is pressed
        self.ix, self.iy = -1, -1  # Initial mouse position

    def draw_rectangle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Start drawing the rectangle (mouse-pressed)
            self.drawing = True
            self.ix, self.iy = x, y
    
            # Save the current state of both the mask and the image for undo purposes
            self.history.append((self.mask.copy(), self.image_cv.copy()))
    
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                # Display the rectangle as we move the mouse (on the image)
                img_copy = self.image_cv.copy()
    
                # Get the minimum and maximum coordinates to handle all rectangle directions
                x_min, x_max = min(self.ix, x), max(self.ix, x)
                y_min, y_max = min(self.iy, y), max(self.iy, y)
    
                # Draw the rectangle on the copy of the image
                cv2.rectangle(img_copy, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)
                cv2.imshow('Image', img_copy)
    
                # Dynamically update the mask while drawing
                mask_copy = self.mask.copy()
                mask_copy[y_min:y_max, x_min:x_max] = 255
                cv2.imshow('Mask', mask_copy)
    
        elif event == cv2.EVENT_LBUTTONUP:
            # Finish drawing the rectangle (mouse released)
            self.drawing = False
    
            # Get the minimum and maximum coordinates to handle all rectangle directions
            x_min, x_max = min(self.ix, x), max(self.ix, x)
            y_min, y_max = min(self.iy, y), max(self.iy, y)
    
            # Draw the rectangle on the original image
            cv2.rectangle(self.image_cv, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)
            cv2.imshow('Image', self.image_cv)
    
            # Update the mask with the final rectangle
            self.mask[y_min:y_max, x_min:x_max] = 255
            cv2.imshow('Mask', self.mask)

    def undo(self):
        if len(self.history) > 0:
            # Revert to the last saved state in history
            self.mask, self.image_cv = self.history.pop()
            cv2.imshow('Mask', self.mask)
            cv2.imshow('Image', self.image_cv)

    def generate_mask(self, screen_width=2550, screen_height=1440):
        self.__init__(self.image)

        # Create a windows and bind the function to the mouse event in the Image window
        cv2.namedWindow('Image')
        cv2.namedWindow('Mask')  # Create a separate window for the mask
        cv2.setMouseCallback('Image', self.draw_rectangle)

        # Wait for a short period to ensure windows are initialized
        cv2.waitKey(250)

        # Position the windows near each other
        image_window_x = int(screen_width * 0.05)
        image_window_y = int(screen_height * 0.1)
        mask_window_x = image_window_x + self.image.width + 100  # 100 pixels space between them
        mask_window_y = image_window_y

        cv2.moveWindow('Image', image_window_x, image_window_y)
        cv2.moveWindow('Mask', mask_window_x, mask_window_y)

        # Main loop
        while True:
            cv2.imshow('Image', self.image_cv)
            cv2.imshow('Mask', self.mask)  # Continuously show the mask

            key = cv2.waitKey(1) & 0xFF

            # Press 'ESC' to exit
            if key == 27:
                break

            # Check for Z
            if key in [ord('z'), ord('Z')]:
                self.undo()

        # Cleanup
        cv2.destroyAllWindows()

        # Convert mask to image
        self.mask = Image.fromarray(self.mask)
