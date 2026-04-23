import joblib
import numpy as np

MODEL_PATH  = "models_saved/model.pkl"
SCALER_PATH = "models_saved/scaler.pkl"

model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def predict(features):
    X = scaler.transform([features])
    proba = model.predict_proba(X)[0]
    label = model.classes_[np.argmax(proba)]
    confidence = float(np.max(proba))
    return {"label": label, "confidence": round(confidence, 3)}