import numpy as np
from numpy.linalg import norm


def SNR(s_clean, s_noisy):
    """
    Signal-to-noise ratio:
    Paper: Evaluation of Objective Quality
    Measures for Speech Enhancement
    https://pdfs.semanticscholar.org/4974/18c70971c8d990e2edf989d6f05675b7c23a.pdf
    :param s_clean: clean signal
    :param s_noisy: noisy signal
    :return: measure of the quality of denoiser
    """
    # normalization:
    n = min(s_clean.shape[0], s_noisy.shape[0])
    s_clean, s_noisy = s_clean[: n], s_noisy[: n]

    # compute the Sound-to-Noise Ratio:
    snr = 20 * np.log10(norm(s_clean) / norm(s_clean - s_noisy))

    return snr

