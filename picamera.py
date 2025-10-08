# pi_capture.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from picamera2 import Picamera2
from threading import Lock
from PIL import Image
import io
import requests
import uvicorn

# PC backend URL (replace with your PC LAN IP)
PC_BACKEND_URL = "http://10.190.160.115:8000/predict/"

app = FastAPI()



# Enable CORS so mobile frontend can access Pi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

camera_lock = Lock()

def capture_image():
    from time import sleep
    from PIL import Image
    import io

    with camera_lock:  # Ensure only one capture at a time
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
        picam2.configure(config)
        picam2.start()
        sleep(0.5)  # Give time for camera to initialize
        image_array = picam2.capture_array()
        picam2.stop()

        image = Image.fromarray(image_array)
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        buf.seek(0)
        return buf



def send_to_pc_backend(image_bytes):
    files = {"file": ("leaf.jpg", image_bytes, "image/jpeg")}
    try:
        response = requests.post(PC_BACKEND_URL, files=files, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"PC backend request failed: {e}"}

@app.get("/capture/")
def capture():
    """
    Capture image from Pi camera and forward to PC backend for prediction.
    Returns disease + insights.
    """
    try:
        img_bytes = capture_image()
        result = send_to_pc_backend(img_bytes)
        # Include a preview image URL (optional, if you serve images from Pi)
        # For simplicity, we convert to base64 string for frontend
        import base64
        image_base64 = base64.b64encode(img_bytes.getvalue()).decode("utf-8")
        result["image_base64"] = f"data:image/jpeg;base64,{image_base64}"
        return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
