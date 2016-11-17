import numpy as np
from scipy import stats
from tsfresh.feature_extraction import feature_calculators


def mean(time_series):
    return np.mean(time_series)


def std(time_series):
    return np.std(time_series)


def min(time_series):
    return np.min(time_series)


def max(time_series):
    return np.max(time_series)


def quantiles(time_series):
    return [
        np.percentile(time_series, 12.5, interpolation='midpoint'),
        np.percentile(time_series, 25, interpolation='midpoint'),
        np.percentile(time_series, 37.5, interpolation='midpoint'),
        np.percentile(time_series, 50, interpolation='midpoint'),
        np.percentile(time_series, 62.5, interpolation='midpoint'),
        np.percentile(time_series, 75, interpolation='midpoint'),
        np.percentile(time_series, 87.5, interpolation='midpoint'),
    ]


def sum(time_series):
    return np.sum(time_series)


def get_basic_stats(time_series):
    return [mean(time_series), std(time_series), min(time_series), max(time_series), sum(time_series)] + \
           quantiles(time_series)


def standard_mean_error(time_series):
    return stats.sem(time_series)


def linear_weighted_average(time_series):
    return np.average(time_series, weights=[i + 1 for i in xrange(len(time_series))])


def quadratic_weighted_average(time_series):
    return np.average(time_series, weights=[(i + 1) ** 2 for i in xrange(len(time_series))])


def arg_max(time_series):
    return np.argmax(time_series) / float(len(time_series))


def arg_min(time_series):
    return np.argmin(time_series) / float(len(time_series))


def skewness(time_series):
    return stats.skew(time_series)


def kurtosis(time_series):
    return stats.kurtosis(time_series)


def mean_absolute_deviation(time_series):
    return np.mean(np.absolute(time_series - np.mean(time_series)))


def median_absolute_deviation(time_series):
    return np.percentile(np.absolute(time_series - np.percentile(time_series, 50)), 50)


def absolute_energy(time_series):
    return feature_calculators.abs_energy(time_series)


def count_above_mean(time_series):
    return feature_calculators.count_above_mean(time_series)


def count_below_mean(time_series):
    return feature_calculators.count_below_mean(time_series)


def large_number_of_peaks(time_series):
    return [
        int(feature_calculators.large_number_of_peaks(time_series, 3)),
        int(feature_calculators.large_number_of_peaks(time_series, 5)),
        int(feature_calculators.large_number_of_peaks(time_series, 10)),
        int(feature_calculators.large_number_of_peaks(time_series, 25)),
        int(feature_calculators.large_number_of_peaks(time_series, 50))
    ]


def longest_strike_above_mean(time_series):
    return feature_calculators.longest_strike_above_mean(time_series)


def longest_strike_below_mean(time_series):
    return feature_calculators.longest_strike_below_mean(time_series)


def mean_autocorrelation(time_series):
    return feature_calculators.mean_autocorrelation(time_series)


def number_peaks(time_series):
    return [
        feature_calculators.number_peaks(time_series, 5),
        feature_calculators.number_peaks(time_series, 10),
        feature_calculators.number_peaks(time_series, 20),
        feature_calculators.number_peaks(time_series, 30),
        feature_calculators.number_peaks(time_series, 50),
    ]


def time_reversal_assymmetry_statistic(time_series):
    return [
        feature_calculators.time_reversal_asymmetry_statistic(time_series, 25),
        feature_calculators.time_reversal_asymmetry_statistic(time_series, 50),
        feature_calculators.time_reversal_asymmetry_statistic(time_series, 100),
        feature_calculators.time_reversal_asymmetry_statistic(time_series, 200),
    ]


def variance(time_series):
    return feature_calculators.variance(time_series)


STATISTICAL_FEATURES = [mean, std, min, max, quantiles, sum, linear_weighted_average, quadratic_weighted_average,
                        arg_max, arg_min, skewness, kurtosis, standard_mean_error,
                        mean_absolute_deviation, median_absolute_deviation,
                        absolute_energy, count_above_mean, count_below_mean, large_number_of_peaks,
                        longest_strike_above_mean, longest_strike_below_mean, mean_autocorrelation,
                        number_peaks, time_reversal_assymmetry_statistic, variance]
