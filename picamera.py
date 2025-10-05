from picamera2 import Picamera2
import time

picam2 = Picamera2()
picam2.start()
time.sleep(2)  # wait for auto-exposure
picam2.capture_file("test.jpg")
print("✅ Captured image as test.jpg")
