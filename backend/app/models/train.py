from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

MODEL_PATH = "backend/models_saved/model.pkl"
SCALER_PATH = "backend/models_saved/scaler.pkl"

def train_model(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_scaled, y)

    os.makedirs("backend/models_saved", exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    return model