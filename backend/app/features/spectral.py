#backend.
from app.dsp.fft import compute_fft
import numpy as np
import librosa


def extract_features(signal, sr):
    freqs, mag = compute_fft(signal, sr)
    
    mag   = mag[:len(mag)//2]
    freqs = freqs[:len(freqs)//2]
    mfcc = librosa.feature.mfcc(y=signal.astype(float), sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)  # 13 valeurs supplémentaires

    # Protection division par zéro
    sum_mag = np.sum(mag)
    if sum_mag == 0:
        return {
            "dominant_freq": 0.0,
            "energy": 0.0,
            "centroid": 0.0,
            "bandwidth": 0.0,
            "zcr": 0.0,
            "rms": 0.0,
            
        }

    dominant_freq = freqs[np.argmax(mag)]
    energy        = np.sum(mag**2)
    centroid      = np.sum(freqs * mag) / sum_mag
    bandwidth     = np.sqrt(np.sum(((freqs - centroid)**2) * mag) / sum_mag)
    zcr           = np.mean(np.abs(np.diff(np.sign(signal))))
    rms           = np.sqrt(np.mean(signal**2))

    return {
        "dominant_freq": float(dominant_freq),
        "energy":        float(energy),
        "centroid":      float(centroid),
        "bandwidth":     float(bandwidth),
        "zcr":           float(zcr),
        "rms":           float(rms),
         **{f"mfcc_{i}": float(v) for i, v in enumerate(mfcc_mean)}
    }