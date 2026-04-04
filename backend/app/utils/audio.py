import librosa
import numpy as np


def load_audio(file, sr=22050):
    """
    Charge un fichier audio UploadFile (FastAPI)
    """
    signal, sample_rate = librosa.load(file.file, sr=sr)
    return signal, sample_rate