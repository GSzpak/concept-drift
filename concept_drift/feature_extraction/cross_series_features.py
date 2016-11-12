import warnings

from scipy import stats


def cross_correlation(time_series1, time_series2):
    assert len(time_series1) == len(time_series2)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('error')
        try:
            correlation, _ = stats.pearsonr(time_series1, time_series2)
            return correlation
        except RuntimeWarning:
            return 0.0
