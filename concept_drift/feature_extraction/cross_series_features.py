import numpy as np
from scipy import stats


def cross_correlation(time_series1, time_series2):
    assert len(time_series1) == len(time_series2)
    correlation, _ = stats.pearsonr(time_series1, time_series2)
    if np.isnan(correlation):
        return 0.0
    return correlation
