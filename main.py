from fastapi import FastAPI, HTTPException, UploadFile, File, Header
from pydantic import BaseModel
from typing import Optional
import cv2
import numpy as np
import requests
from tensorflow.keras.models import load_model
import os
import hashlib
import firebase_admin
from firebase_admin import credentials, auth, firestore
from middleware import setup_cors
from datetime import datetime

app = FastAPI()
setup_cors(app)

# ---------------- Model info ----------------
MODEL_URL = "https://github.com/Saad-Iqbal-tech/BrainTumorAPI/releases/download/Tag/vgg16_brain_model.h5"
MODEL_FILE = "vgg16_brain_model.h5"
MODEL_SHA256 = "96b84020c0dbe3bff47d5ced16ab99e277978023d1e7f41c8c83e78565b22f52"
IMG_SIZE = (256, 256)

# ---------------- Firebase Init ----------------
cred = credentials.Certificate(os.environ.get("FIREBASE_ADMIN_KEY_PATH"))
firebase_admin.initialize_app(cred)
db = firestore.client()

class_names = ["notumor", "tumor"]

# ---------------- Profile Model (FIXED) ----------------
class Profile(BaseModel):
    name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    phone: Optional[str] = None

# ---------------- Utils ----------------
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
    if r.status_code != 200:
        raise RuntimeError("Failed to download model")

    with open(MODEL_FILE, "wb") as f:
        f.write(r.content)

    if file_sha256(MODEL_FILE) != MODEL_SHA256:
        raise RuntimeError("Model file corrupted")

download_model()
model = load_model(MODEL_FILE)

# ---------------- Image Preprocessing ----------------
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

    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

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
    brain_crop = cv2.cvtColor(brain_crop, cv2.COLOR_GRAY2RGB)
    brain_crop = brain_crop.astype(np.float32) / 255.0
    brain_crop = np.expand_dims(brain_crop, axis=0)

    return brain_crop

# ---------------- Auth Utils (FIXED) ----------------
def verify_token(id_token: str):
    try:
        decoded = auth.verify_id_token(id_token)
        return decoded["uid"]
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid authentication token")

def get_uid_from_header(authorization: Optional[str]):
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    token = authorization.replace("Bearer ", "")
    return verify_token(token)

# ---------------- Profile Endpoints ----------------
@app.post("/profile")
async def save_profile(
    profile: Profile,
    authorization: Optional[str] = Header(None)
):
    uid = get_uid_from_header(authorization)

    db.collection("users").document(uid).set({
        "name": profile.name,
        "age": profile.age,
        "gender": profile.gender,
        "phone": profile.phone,
        "updated_at": firestore.SERVER_TIMESTAMP
    }, merge=True)

    return {"message": "Profile saved successfully!"}

@app.get("/profile")
async def get_profile(authorization: Optional[str] = Header(None)):
    uid = get_uid_from_header(authorization)

    doc = db.collection("users").document(uid).get()

    if not doc.exists:
        return {"exists": False, "data": None}

    return {"exists": True, "data": doc.to_dict()}

# ---------------- Prediction Endpoint ----------------
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(None)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image")

    uid = get_uid_from_header(authorization)

    content = await file.read()
    img_preprocessed = preprocess_image(content)

    preds = model.predict(img_preprocessed)[0][0]
    pred_class = "tumor" if preds >= 0.5 else "notumor"
    confidence = float(preds) if pred_class == "tumor" else 1 - float(preds)

    doc = db.collection("users").document(uid).get()
    patient_name = doc.to_dict().get("name", "Unknown") if doc.exists else "Unknown"

    db.collection("reports").add({
        "user_id": uid,
        "patient_name": patient_name,
        "prediction": pred_class,
        "confidence": confidence,
        "created_at": firestore.SERVER_TIMESTAMP
    })

    return {
        "user_id": uid,
        "predicted_class": pred_class,
        "confidence": confidence,
        "patient_name": patient_name
    }

# ---------------- Reports Endpoint ----------------
@app.get("/reports")
async def get_reports(authorization: Optional[str] = Header(None)):
    uid = get_uid_from_header(authorization)

    reports_ref = (
        db.collection("reports")
        .where("user_id", "==", uid)
        .order_by("created_at", direction=firestore.Query.DESCENDING)
    )

    reports = [doc.to_dict() for doc in reports_ref.stream()]
    return {"reports": reports}

