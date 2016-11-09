import numpy as np
from scipy import stats

from concept_drift.feature_extraction.cross_series_features import cross_correlation


def mean(time_series):
    return np.mean(time_series)


def standard_mean_error(time_series):
    return stats.sem(time_series)


def std(time_series):
    return np.std(time_series)


def min(time_series):
    return np.min(time_series)


def max(time_series):
    return np.max(time_series)


def quantiles(time_series):
    return [
        np.percentile(time_series, 10, interpolation='midpoint'),
        np.percentile(time_series, 25, interpolation='midpoint'),
        np.percentile(time_series, 50, interpolation='midpoint'),
        np.percentile(time_series, 75, interpolation='midpoint'),
        np.percentile(time_series, 90, interpolation='midpoint'),
    ]


def linear_weighted_average(time_series):
    return np.average(time_series, weights=[i + 1 for i in xrange(len(time_series))])


def quadratic_weighted_average(time_series):
    return np.average(time_series, weights=[(i + 1) ** 2 for i in xrange(len(time_series))])


def arg_max(time_series):
    return np.argmax(time_series) / float(len(time_series))


def arg_min(time_series):
    return np.argmin(time_series) / float(len(time_series))


def kurtosis(time_series):
    return stats.kurtosis(time_series)


def mean_absolute_deviation(time_series):
    return np.mean(np.absolute(time_series - np.mean(time_series)))


def median_absolute_deviation(time_series):
    return np.percentile(np.absolute(time_series - np.percentile(time_series, 50)), 50)


def autocorrelation(time_series):
    return cross_correlation(time_series, time_series)


BASIC_STATS = [mean, std, min, max, quantiles]


STATISTICAL_FEATURES = [mean, std, min, max, quantiles, linear_weighted_average, quadratic_weighted_average,
                        arg_max, arg_min, kurtosis, standard_mean_error,
                        mean_absolute_deviation, median_absolute_deviation]
