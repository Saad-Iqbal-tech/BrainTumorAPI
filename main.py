from fastapi import HTTPException, FastAPI, UploadFile, File
import firebase
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from pydantic import BaseModel
from firebase_admin import auth
import io
from PIL import Image
import numpy as np
import requests
from middleware import setup_cors

app = FastAPI()
setup_cors(app)


class UserSignup(BaseModel):
    email: str
    password: str
    confirm_password: str

class UserLogin(BaseModel):
    email: str
    password: str

MODEL_PATH = "vgg16_brain_model.h5"
model = load_model(MODEL_PATH)
class_names = ["notumor", "tumor"]
IMG_SIZE = (256, 256)

def preprocess_image(file_bytes):
    file_bytes = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not read image")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    blur = cv2.GaussianBlur(gray, (5,5), 0)
    blur = cv2.medianBlur(blur, 5)

    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest], -1, 255, -1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
        brain = cv2.bitwise_and(gray, mask)
        coords = cv2.findNonZero(brain)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            brain_crop = brain[y:y+h, x:x+w]
        else:
            brain_crop = gray
    else:
        brain_crop = gray

    brain_crop = cv2.resize(brain_crop, IMG_SIZE)

    brain_crop = brain_crop.astype(np.float32)

    brain_crop = cv2.cvtColor(brain_crop, cv2.COLOR_GRAY2RGB)

    brain_crop = brain_crop / 255.0

    brain_crop = np.expand_dims(brain_crop, axis=0)  # (1, 256, 256, 3)

    return brain_crop

@app.post("/signup/")
def signup(data: UserSignup):
    try:
        user = auth.create_user(email=data.email, password=data.password,confirm_password=data.confirm_password)
        return {"message": "User created successfully", "uid": user.uid}
    except Exception as e:
        return HTTPException(status_code=400, detail=str(e))

@app.post("/login/")
def login(data: UserLogin):
    url = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key=AIzaSyAr1GVSoXOSeqD3IylCHKK_jQRKH8yQGU4"
    payload = {"email": data.email, "password": data.password, "returnSecureToken": True}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        return HTTPException(status_code=400, detail="Invalid credentials")

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith('image/'):
            return HTTPException(status_code=400, detail="File is not an image")
        return {"message": "Image uploaded"}
    except Exception as e:
        return HTTPException(status_code=400, detail=str(e))

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image")
    
    content = await file.read()
    try:
        img_preprocessed = preprocess_image(content)
        preds = model.predict(img_preprocessed)[0][0]  # Single sigmoid output

        pred_class = "tumor" if preds >= 0.5 else "notumor"
        confidence = float(preds) if pred_class == "tumor" else 1 - float(preds)

        return {
            "predicted_class": pred_class,
            "confidence": confidence
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



