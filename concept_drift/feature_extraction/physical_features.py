import numpy as np
import pywt

from concept_drift.feature_extraction.statistical_features import get_basic_stats


TRANSFORM_FEATURES_NUM = 5


def dft_features(time_series):
    dft_coeff = np.fft.fft(time_series)
    # Second half of coefficients is conjugate to first half
    dft_coeff = dft_coeff[:len(dft_coeff) / 2]
    dft_coeff_selected = sorted(dft_coeff, key=lambda elem: np.absolute(elem), reverse=True)[:TRANSFORM_FEATURES_NUM]
    return list(np.real(dft_coeff_selected)) + list(np.imag(dft_coeff_selected)) + \
           get_basic_stats(np.real(dft_coeff)) + get_basic_stats(np.imag(dft_coeff))


def dwt_features(time_series):
    approximation_coefficients, detail_coefficients = pywt.dwt(time_series, 'haar')
    approximation_coeff_selected = sorted(approximation_coefficients, reverse=True)[:TRANSFORM_FEATURES_NUM]
    detail_coeff_selected = sorted(detail_coefficients, reverse=True)[:TRANSFORM_FEATURES_NUM]
    return approximation_coeff_selected + detail_coeff_selected + \
           get_basic_stats(approximation_coefficients) + get_basic_stats(detail_coefficients)


PHYSICAL_FEATURES = [dft_features, dwt_features]
