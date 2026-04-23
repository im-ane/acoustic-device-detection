from app.utils.audio import load_audio
from app.dsp.filter import low_pass_filter
from app.dsp.segmentation import split_signal
from app.features.spectral import extract_features
from app.models.predict import predict
import numpy as np

def analyze_audio(file):

    # 1. Charger et filtrer
    signal, sr = load_audio(file)
    signal_filtered = low_pass_filter(signal)
    segments = split_signal(signal_filtered, sr)

    # 2. Extraire features par segment
    results = []
    for seg in segments:
        f = extract_features(seg, sr)
        results.append(f)

    # 3. Vecteur moyen sur tous les segments → prédiction
    arrays = [list(f.values()) for f in results]
    mean_vector = np.nan_to_num(np.mean(arrays, axis=0), nan=0.0)
    device = predict(mean_vector)

    return {
        "device"  : device,
        "segments": len(segments),
        "features": results[0] if results else {}
    }