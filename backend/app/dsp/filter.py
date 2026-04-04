from scipy.signal import butter, lfilter


def low_pass_filter(signal, cutoff=3000, fs=22050, order=5):
    """
    Filtre passe-bas (enlève hautes fréquences)
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist

    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_signal = lfilter(b, a, signal)

    return filtered_signal