import os
import pandas as pd
from app.core.config import RAW_DATA_DIR

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