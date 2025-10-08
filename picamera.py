# pi_capture.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from picamera2 import Picamera2
from PIL import Image
import io
import requests
import uvicorn

# Your backend API URL (replace with your PC's LAN IP)
BACKEND_URL = "http://10.190.160.115:8000/predict/"

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

def capture_image():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()
    image_array = picam2.capture_array()
    picam2.stop()

    image = Image.fromarray(image_array)
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)
    return buf

def send_to_backend(image_bytes):
    files = {"file": ("leaf.jpg", image_bytes, "image/jpeg")}
    response = requests.post(BACKEND_URL, files=files)
    return response.json()

@app.get("/capture/")
def capture():
    """
    Capture image from Pi camera and forward to backend for prediction.
    Returns backend response (disease + insights).
    """
    try:
        img_bytes = capture_image()
        result = send_to_backend(img_bytes)
        return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    # Start FastAPI server when script runs
    uvicorn.run(app, host="0.0.0.0", port=5000)
