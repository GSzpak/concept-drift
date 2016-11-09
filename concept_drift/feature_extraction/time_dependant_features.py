import numpy as np


def avg_derivatives(time_series, time):
    def derivative(sequence):
        return [sequence[i] - sequence[i - 1] for i in xrange(1, len(sequence))]
    first_derivative = derivative(time_series)
    second_derivative = derivative(first_derivative)
    return {
        'first_derivative_avg': np.mean(first_derivative),
        'second_derivative_avg': np.mean(second_derivative)
    }


def avg_integrals(time_series):
    def integral(sequence):
        return [(sequence[i] + sequence[i - 1]) / 2 for i in xrange(1, len(sequence))]
    first_integral = integral(time_series)
    second_integral = integral(first_integral)
    return {
        'first_integral_avg': sum(first_integral),
        'second_integral_avg': sum(second_integral)
    }