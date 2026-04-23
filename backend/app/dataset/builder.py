import os
import numpy as np
import librosa
from backend.app.core.config import RAW_DATA_DIR, DATA_DIR,PROCESSED_DIR
from backend.app.dsp.filter import low_pass_filter
from backend.app.dsp.segmentation import split_signal
from backend.app.features.spectral import extract_features


def load_raw():
    """Charge tous les .wav depuis data/raw/label/ → retourne liste (path, label)."""
    data = []
    for label in os.listdir(PROCESSED_DIR):  # ← lit processed/ sans clusters
        folder = os.path.join(PROCESSED_DIR, label)
        if not os.path.isdir(folder):
            continue
        for f in os.listdir(folder):
            if f.lower().endswith(".wav"):
                data.append((os.path.join(folder, f), label))
    return data


def extract_features_file(file_path):
    """Pipeline DSP complet sur un fichier → vecteur moyen."""
    signal, sr = librosa.load(file_path, sr=22050, mono=True)
    segments = split_signal(low_pass_filter(signal), sr)
    arrays = [list(extract_features(seg, sr).values()) for seg in segments]
    result = np.mean(arrays, axis=0)
    return np.nan_to_num(result, nan=0.0)


def build_dataset():
    """Construit X, y depuis data/raw/ et sauvegarde dans data/datasets/."""
    all_data = load_raw()
    X, y = [], []

    for i, (path, label) in enumerate(all_data):
        try:
            X.append(extract_features_file(path))
            y.append(label)
        except Exception as e:
            print(f"⚠️  Ignoré {os.path.basename(path)}: {e}")
        if (i+1) % 50 == 0:
            print(f"   {i+1}/{len(all_data)}...")

    X, y = np.array(X), np.array(y)

    out = os.path.join(DATA_DIR, "datasets")
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "X.npy"), X)
    np.save(os.path.join(out, "y.npy"), y)
    print(f"✅ Dataset : {X.shape[0]} samples, {len(set(y))} classes")
    return X, y


def load_dataset():
    """Charge un dataset déjà construit."""
    path = os.path.join(DATA_DIR, "datasets")
    return np.load(os.path.join(path, "X.npy")), \
           np.load(os.path.join(path, "y.npy"), allow_pickle=True)