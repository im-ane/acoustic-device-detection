import numpy as np


def split_signal(signal, sr, window_size=1.0, overlap=0.5):
    """
    Découpe le signal en segments

    window_size: durée en secondes
    overlap: chevauchement (0 à 1)
    """

    window_length = int(window_size * sr)
    step = int(window_length * (1 - overlap))

    segments = []

    for start in range(0, len(signal) - window_length, step):
        segment = signal[start:start + window_length]
        segments.append(segment)

    return segments