
from app.dsp.fft import compute_fft
import numpy as np


def extract_features(signal, sr):
    freqs, mag = compute_fft(signal, sr)
    
    mag = mag[:len(mag)//2]
    freqs = freqs[:len(freqs)//2]

    dominant_freq = freqs[np.argmax(mag)]
    energy = np.sum(mag**2)
    centroid = np.sum(freqs * mag) / np.sum(mag)

    bandwidth = np.sqrt(
        np.sum(((freqs - centroid) ** 2) * mag) / np.sum(mag)
    )

    zcr = np.mean(np.abs(np.diff(np.sign(signal))))

    rms = np.sqrt(np.mean(signal**2))
    return {
        "dominant_freq": float(dominant_freq),
        "energy": float(energy),
        "centroid": centroid,
        "bandwidth": bandwidth,
        "zcr": zcr,
        "rms":rms
    }
