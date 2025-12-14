from fastapi import FastAPI, HTTPException, UploadFile, File, Header
import cv2
import numpy as np
import requests
from tensorflow.keras.models import load_model
from middleware import setup_cors
import os
import hashlib
import firebase_admin
from firebase_admin import credentials, auth, firestore
from datetime import datetime

app = FastAPI()
setup_cors(app)

MODEL_URL = "https://github.com/Saad-Iqbal-tech/BrainTumorAPI/releases/download/Tag/vgg16_brain_model.h5"
MODEL_FILE = "vgg16_brain_model.h5"
MODEL_SHA256 = "96b84020c0dbe3bff47d5ced16ab99e277978023d1e7f41c8c83e78565b22f52"
IMG_SIZE = (256, 256)
class_names = ["notumor", "tumor"]

# Initialize Firebase
cred = credentials.Certificate(os.environ.get("FIREBASE_ADMIN_KEY_PATH"))
firebase_admin.initialize_app(cred)
db = firestore.client()  # Firestore for users & reports

def file_sha256(path):
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        sha.update(f.read())
    return sha.hexdigest()

def download_model():
    if os.path.exists(MODEL_FILE):
        if file_sha256(MODEL_FILE) == MODEL_SHA256:
            return
        os.remove(MODEL_FILE)
    r = requests.get(MODEL_URL)
    if r.status_code == 200:
        with open(MODEL_FILE, "wb") as f:
            f.write(r.content)
    else:
        raise RuntimeError("Failed to download model")
    if file_sha256(MODEL_FILE) != MODEL_SHA256:
        raise RuntimeError("Model file corrupted")

download_model()
model = load_model(MODEL_FILE)

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
    brain_crop = np.expand_dims(brain_crop, axis=0)
    return brain_crop

def verify_token(id_token: str):
    """Verify Firebase token and return UID"""
    try:
        decoded_token = auth.verify_id_token(id_token)
        return decoded_token["uid"]
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid authentication token")

# --- Profile endpoints ---
@app.post("/profile")
def save_profile(data: dict, authorization: str = Header(...)):
    """Save dashboard form data to Firestore"""
    token = authorization.replace("Bearer ", "")
    uid = verify_token(token)
    # Save in users collection
    db.collection("users").document(uid).set({
        "name": data.get("name", ""),
        "age": data.get("age", ""),
        "gender": data.get("gender", ""),
        "phone": data.get("phone", "")
    })
    return {"message": "Profile saved successfully"}

@app.get("/profile")
def get_profile(authorization: str = Header(...)):
    """Return profile info for dashboard form"""
    token = authorization.replace("Bearer ", "")
    uid = verify_token(token)
    doc = db.collection("users").document(uid).get()
    if not doc.exists:
        return {"exists": False}
    return {"exists": True, "data": doc.to_dict()}

# --- Prediction endpoint ---
@app.post("/predict")
async def predict(file: UploadFile = File(...), authorization: str = Header(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image")

    token = authorization.replace("Bearer ", "")
    uid = verify_token(token)

    # Preprocess and predict
    content = await file.read()
    img_preprocessed = preprocess_image(content)
    preds = model.predict(img_preprocessed)[0][0]
    pred_class = "tumor" if preds >= 0.5 else "notumor"
    confidence = float(preds) if pred_class == "tumor" else 1 - float(preds)

    # Get user name for report
    user_doc = db.collection("users").document(uid).get()
    if not user_doc.exists:
        raise HTTPException(status_code=400, detail="User profile not completed")
    user_data = user_doc.to_dict()
    patient_name = user_data.get("name", "Unknown")

    # Save report in Firestore
    report_data = {
        "user_id": uid,
        "patient_name": patient_name,
        "prediction": pred_class,
        "confidence": confidence,
        "created_at": datetime.utcnow().isoformat()
    }
    db.collection("reports").add(report_data)

    return {
        "patient_name": patient_name,
        "predicted_class": pred_class,
        "confidence": confidence
    }

