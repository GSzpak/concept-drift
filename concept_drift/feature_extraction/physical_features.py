import numpy as np
import pywt

from concept_drift.feature_extraction.statistical_features import get_basic_stats
from tsfresh.feature_extraction import feature_calculators


TRANSFORM_FEATURES_NUM = 10


def _get_coefficients_stats(coefficients, num_to_select):
    return list(coefficients[:num_to_select]) + get_basic_stats(coefficients)


def get_fft_coeffs(time_series):
    fft_coeffs = feature_calculators.fft_coefficient(
        time_series,
        '',
        [{'coeff': i} for i in xrange(len(time_series))]
    )
    fft_coeffs = fft_coeffs.values
    fft_coeffs = np.trim_zeros(fft_coeffs)
    assert len(fft_coeffs) == (len(time_series) / 2) + 1 \
        if len(time_series) % 2 == 0 \
        else (len(time_series) + 1) / 2
    return fft_coeffs


def dft_features(time_series):
    dft_coeff = get_fft_coeffs(time_series)
    return _get_coefficients_stats(dft_coeff, TRANSFORM_FEATURES_NUM)


def dwt_features(time_series):
    approximation_coefficients, detail_coefficients = pywt.dwt(time_series, 'haar')
    return _get_coefficients_stats(approximation_coefficients, TRANSFORM_FEATURES_NUM / 2) + \
        _get_coefficients_stats(detail_coefficients, TRANSFORM_FEATURES_NUM / 2)


PHYSICAL_FEATURES = [dft_features, dwt_features]
