from picamera2 import Picamera2
from PIL import Image

picam2 = Picamera2()
picam2.start()

# Capture the image as a NumPy array
image = picam2.capture_array()

# Convert RGBA/other mode to RGB before saving as JPEG
img = Image.fromarray(image).convert("RGB")
img.save("test.jpg")

print("Image saved as test.jpg")

picam2.stop()
