import os
import pandas as pd
from backend.app.core.config import RAW_DATA_DIR
from backend.app.core.config import DATA_DIR
import numpy as np

ESC50_PATH = os.path.join(RAW_DATA_DIR, "esc50")

TARGET_CLASSES = [
    "engine",
    "air_conditioner",
    "vacuum_cleaner",
    "washing_machine"
]

def load_esc50():
    csv_path = os.path.join(ESC50_PATH, "meta", "esc50.csv")
    audio_path = os.path.join(ESC50_PATH, "audio")

    df = pd.read_csv(csv_path)
    df = df[df["category"].isin(TARGET_CLASSES)]

    data = []
    for _, row in df.iterrows():
        file_path = os.path.join(audio_path, row["filename"])
        data.append((file_path, row["category"]))

    return data


def load_personal():
    base_path = os.path.join(RAW_DATA_DIR, "personal")

    data = []
    for label in os.listdir(base_path):
        folder = os.path.join(base_path, label)

        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            data.append((path, label))

    return data

def load_dataset():
    path = os.path.join(DATA_DIR, "datasets")

    X = np.load(os.path.join(path, "X.npy"))
    y = np.load(os.path.join(path, "y.npy"))

    return X, y

def build_dataset():
    esc50_data = load_esc50()
    personal_data = load_personal()

    # Combine datasets and preprocess
    # ... (preprocessing logic)

    return esc50_data,personal_data
