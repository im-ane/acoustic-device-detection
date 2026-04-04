import joblib

MODEL_PATH = "backend/models_saved/model.pkl"
SCALER_PATH = "backend/models_saved/scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def predict(features):
    X = scaler.transform([features])
    return model.predict(X)[0]