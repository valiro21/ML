import numpy


def self_dot(array):
    return numpy.dot(array, array)

def _first_max_index(values):
    max_found = False
    max_value = 0
    max_index = 0
    index = 0
    for value in values:
        if (not max_found or value > max_value):
            max_value = value
            max_index = index
            max_found = True
        index += 1

    return max_index

class Functions:
    sigmoid = lambda x: 1.0 / (1.0 + numpy.exp(-numpy.array(x)))

    mse = lambda y, a: self_dot([value[0] - value[1] for value in zip(y, a)]) / 2.0

    first_max_neuron = lambda values: _first_max_index(values)

    no_change = lambda values: values

    all = lambda values: values if isinstance(values, bool) else all(values)


class FunctionsDerivative:
    sigmoid = lambda x: -(Functions.sigmoid(x) - 1) * Functions.sigmoid(x)
    mse_by_activation_derivative = lambda a, y, z, activation_derivative: (a - y) * activation_derivative(z)
