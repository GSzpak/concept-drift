from concept_drift.feature_extraction.statistical_features import BASIC_STATS


def derivative(time_series, time):
    assert len(time_series) == len(time)
    return [(time_series[i] - time_series[i - 1]) / float(time[i] - time[i - 1])
            for i in xrange(1, len(time_series))]


def integral(time_series, time):
    assert len(time_series) == len(time)
    return [((time_series[i] + time_series[i - 1]) * (time[i] - time[i - 1])) / 2.
            for i in xrange(1, len(time_series))]


def time_related_features(time_series, time):
    der = derivative(time_series, time)
    integr = integral(time_series, time)
    return [stats_fun(der) for stats_fun in BASIC_STATS] + \
           [stats_fun(integr) for stats_fun in BASIC_STATS]
