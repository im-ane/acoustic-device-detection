import numpy as np



def compute_fft(signal, sr):
    fft = np.fft.fft(signal)
    mag = np.abs(fft)

    freqs = np.fft.fftfreq(len(signal), d=1/sr)

    # garder moitié positive
    mag = mag[:len(mag)//2]
    freqs = freqs[:len(freqs)//2]

    return freqs, mag