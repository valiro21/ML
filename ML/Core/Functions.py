import numpy


def self_dot(array):
    return numpy.dot(array, array)


class Functions:
    sigmoid = lambda x: 1.0 / (1.0 + numpy.exp(-numpy.array(x)))

    mse = lambda y, a: self_dot([value[0] - value[1] for value in zip(y, a)]) / 2.0

    first_max_neuron = lambda values: values.index(max(values))

    no_change = lambda values: values


class FunctionsDerivative:
    sigmoid = lambda x: -(Functions.sigmoid(x) - 1) * Functions.sigmoid(x)
    mse_by_activation_derivative = lambda a, y, z, activation_derivative: (a - y) * activation_derivative(z)
