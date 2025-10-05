from picamera2 import Picamera2
from time import sleep

picam2 = Picamera2()
picam2.start_preview()  # optional
sleep(2)  # allow camera to adjust
image = picam2.capture_array()
