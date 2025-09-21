# # backend/leaf_api.py

# import io
# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# from fastapi import FastAPI, File, UploadFile, HTTPException
# from PIL import Image
# import requests

# # --- Initialize FastAPI ---
# app = FastAPI(title="Leaf Disease Detection API")

# # --- Device configuration ---
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # --- Classes (same as training) ---
# classes = [
#     'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
#     'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
#     'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
#     'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
#     'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
#     'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
#     'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
#     'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
#     'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
#     'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
#     'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
#     'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
# ]

# # --- Load the trained MobileNet model ---
# model = models.mobilenet_v2(num_classes=len(classes))
# model.load_state_dict(torch.load("saved_models/leaf_mobilenet.pth", map_location=device))
# model.to(device)
# model.eval()

# # --- Image preprocessing ---
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])
# ])

# # --- Prediction function ---
# def predict_image(image_bytes):
#     image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#     image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
#     with torch.no_grad():
#         outputs = model(image)
#         _, predicted = torch.max(outputs, 1)
#     return classes[predicted.item()]

# # --- Gemini API call ---
# GEMINI_API_KEY = "AIzaSyDGVCQeKgReVYRuuY28eX-JfFuIlkdpzKk"  # Replace with your free API key

# def get_gemini_insights(disease_name: str) -> str:
#     url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
#     headers = {
#         "Content-Type": "application/json",
#         "X-goog-api-key": GEMINI_API_KEY
#     }
#     payload = {
#         "contents": [
#             {
#                 "parts": [
#                     {"text": f"Provide treatment and care tips for the following leaf disease: {disease_name}"}
#                 ]
#             }
#         ]
#     }

#     try:
#         response = requests.post(url, headers=headers, json=payload)
#         response.raise_for_status()
#         data = response.json()
#     except requests.exceptions.RequestException as e:
#         return f"Gemini API request failed: {e}"

#     # Correct path to extract the text
#     try:
#         insights_text = data['candidates'][0]['content']['parts'][0]['text']
#         return insights_text
#     except (KeyError, IndexError):
#         return "No insights generated from Gemini API."



# # --- API route ---
# @app.post("/predict/")
# async def predict(file: UploadFile = File(...)):
#     contents = await file.read()
#     disease = predict_image(contents)
#     insights = get_gemini_insights(disease)
#     return {"disease": disease, "insights": insights}

# # --- Root route ---
# @app.get("/")
# def root():
#     return {"message": "Leaf Disease Detection API is running with Gemini AI!"}



# backend/leaf_api.py

import io
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
import requests

# Raspberry Pi camera
try:
    from picamera2 import Picamera2
    picam = Picamera2()
    picam.configure(picam.create_still_configuration())
except ImportError:
    picam = None
    import cv2

# --- Initialize FastAPI ---
app = FastAPI(title="Leaf Disease Detection API")

# --- Device configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Classes (same as training) ---
classes = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# --- Load the trained MobileNet model ---
model = models.mobilenet_v2(num_classes=len(classes))
model.load_state_dict(torch.load("saved_models/leaf_mobilenet.pth", map_location=device))
model.to(device)
model.eval()

# --- Image preprocessing ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- Prediction function ---
def predict_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return classes[predicted.item()]

# --- Gemini API call ---
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"  # Replace with your real key

def get_gemini_insights(disease_name: str) -> str:
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": GEMINI_API_KEY
    }
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": f"Provide treatment and care tips for the following leaf disease: {disease_name}"}
                ]
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        return f"Gemini API request failed: {e}"

    try:
        insights_text = data['candidates'][0]['content']['parts'][0]['text']
        return insights_text
    except (KeyError, IndexError):
        return "No insights generated from Gemini API."

# --- API route: Upload & Predict ---
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    disease = predict_image(contents)
    insights = get_gemini_insights(disease)
    return {"disease": disease, "insights": insights}

# --- API route: Capture from Raspberry Pi Camera ---
@app.get("/capture/")
def capture_image():
    save_path = "static/captured.jpg"
    os.makedirs("static", exist_ok=True)

    if picam:  # Use PiCamera2
        picam.start()
        frame = picam.capture_array()
        Image.fromarray(frame).save(save_path)
        picam.stop()
    else:  # Use OpenCV fallback
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return {"error": "Failed to capture image"}
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(save_path)

    # Run prediction
    with open(save_path, "rb") as f:
        contents = f.read()
    disease = predict_image(contents)
    insights = get_gemini_insights(disease)

    return {
        "disease": disease,
        "insights": insights,
        "image_url": f"/{save_path}"
    }

# --- Root route ---
@app.get("/")
def root():
    return {"message": "Leaf Disease Detection API is running with Gemini AI!"}

















# backend/leaf_api.py

import io
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
import requests

# --- Raspberry Pi Camera ---
from picamera2 import Picamera2

picam = Picamera2()
picam.configure(picam.create_still_configuration())

# --- Initialize FastAPI ---
app = FastAPI(title="Leaf Disease Detection API")

# --- Device configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Classes (same as training) ---
classes = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# --- Load the trained MobileNet model ---
model = models.mobilenet_v2(num_classes=len(classes))
model.load_state_dict(torch.load("saved_models/leaf_mobilenet.pth", map_location=device))
model.to(device)
model.eval()

# --- Image preprocessing ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- Prediction function ---
def predict_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return classes[predicted.item()]

# --- Gemini API call ---
GEMINI_API_KEY = "AIzaSyDGVCQeKgReVYRuuY28eX-JfFuIlkdpzKk"  # Replace with your real key

def get_gemini_insights(disease_name: str) -> str:
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": GEMINI_API_KEY
    }
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": f"Provide treatment and care tips for the following leaf disease: {disease_name}"}
                ]
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        return f"Gemini API request failed: {e}"

    try:
        insights_text = data['candidates'][0]['content']['parts'][0]['text']
        return insights_text
    except (KeyError, IndexError):
        return "No insights generated from Gemini API."

# --- API route: Upload & Predict ---
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    disease = predict_image(contents)
    insights = get_gemini_insights(disease)
    return {"disease": disease, "insights": insights}

# --- API route: Capture from Raspberry Pi Camera ---
@app.get("/capture/")
def capture_image():
    save_path = "static/captured.jpg"
    os.makedirs("static", exist_ok=True)

    picam.start()
    frame = picam.capture_array()
    Image.fromarray(frame).save(save_path)
    picam.stop()

    # Run prediction
    with open(save_path, "rb") as f:
        contents = f.read()
    disease = predict_image(contents)
    insights = get_gemini_insights(disease)

    return {
        "disease": disease,
        "insights": insights,
        "image_url": f"/{save_path}"
    }

# --- Root route ---
@app.get("/")
def root():
    return {"message": "Leaf Disease Detection API is running with Gemini AI!"}
