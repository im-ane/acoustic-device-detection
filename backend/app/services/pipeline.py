from backend.app.utils.audio import load_audio
from backend.app.dsp.filter import low_pass_filter
from backend.app.dsp.segmentation import split_signal
import numpy as np
from backend.app.features.spectral import extract_features

def analyze_audio(file):

    signal, sr = load_audio(file)
    signal_filtred=low_pass_filter(signal)
    segments = split_signal(signal_filtred,sr)

    results = []

    for seg in segments:
        f = extract_features(seg, sr)  # appel
        results.append(f)

    return results

