from app.dataset.builder import build_dataset
from app.models.train import train_model
from backend.app.dataset.builder import load_dataset

if __name__ == "__main__":
    X, y = build_dataset()
    X, y = load_dataset()

    model = train_model(X, y)

    print("Model trained successfully")