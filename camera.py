from picamera2 import Picamera2, Preview
import time
import numpy as np
import cv2

def is_image_sufficiently_illuminated(image, threshold=50):
    """ Check if the image is sufficiently illuminated.
        This function calculates the average brightness of the image and 
        compares it to a threshold.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    average_brightness = np.mean(gray_image)
    return average_brightness > threshold

def capture_if_illuminated(picam2, filename):
    """ Capture an image if there is sufficient light. """
    # Capture a preview frame
    preview_frame = picam2.capture_array()
    
    # Check if the image is sufficiently illuminated
    if is_image_sufficiently_illuminated(preview_frame):
        print("Sufficient light detected. Capturing image...")
        picam2.capture_file(filename)
        print(f"Image captured and saved as {filename}")
    else:
        print("Insufficient light detected. No image captured.")

# Initialize the camera
picam2 = Picamera2()
camera_config = picam2.create_preview_configuration()
picam2.configure(camera_config)
picam2.start_preview(Preview.QTGL)
picam2.start()

# Wait for 2 seconds to allow for proper preview setup
time.sleep(2)

# Capture the image if sufficient light is detected
capture_if_illuminated(picam2, "test.jpg")

# Stop the preview
picam2.stop_preview()

