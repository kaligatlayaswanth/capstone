

from picamera2 import Picamera2
picam2 = Picamera2()
picam2.start()
image = picam2.capture_array()
print(image.shape)

