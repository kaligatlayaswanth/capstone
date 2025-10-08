from picamera2 import Picamera2

# Initialize the camera
picam2 = Picamera2()

# Start the camera
picam2.start()

# Capture an image as a numpy array
image = picam2.capture_array()

# Save to a file without using OpenCV
from PIL import Image
img = Image.fromarray(image)
img.save("test.jpg")

print("Image saved as test.jpg")

# Stop the camera
picam2.stop()
