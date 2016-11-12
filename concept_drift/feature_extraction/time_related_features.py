from concept_drift.feature_extraction.statistical_features import get_basic_stats


def derivative(time_series, time):
    assert len(time_series) == len(time)
    return [(time_series[i] - time_series[i - 1]) / float(time[i] - time[i - 1])
            for i in xrange(1, len(time_series))]


def integral(time_series, time):
    assert len(time_series) == len(time)
    return [((time_series[i] + time_series[i - 1]) * (time[i] - time[i - 1])) / 2.
            for i in xrange(1, len(time_series))]


def get_time_related_features(time_series, time):
    der = derivative(time_series, time)
    integr = integral(time_series, time)
    return get_basic_stats(der) + get_basic_stats(integr)
