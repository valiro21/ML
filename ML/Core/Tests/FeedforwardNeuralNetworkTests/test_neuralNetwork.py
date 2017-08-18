import multiprocessing
from unittest import TestCase

import numpy
from ML.Core import Functions, FunctionsDerivative, FeedforwardNeuralNetwork

def testAlmostEquals(l1, l2):
    if len(l1) != len(l2):
        return False

    for i in range(len(l1)):
        if isinstance(l1[i], list) and isinstance(l2[i], list):
            if not testAlmostEquals(l1[i], l2[i]):
                return False
        else:
            if not all(numpy.isclose([l1[i]], [l2[i]])):
                return False
    return True


def read_iris_dataset(file):
    with open(file) as f:
        raw_dataset = f.read().splitlines()

        dataset = []
        for row in raw_dataset:
            values = row.split(",")
            input = list(map(float, values[0:4]))
            raw_output = values[4]
            output = [0, 0, 0]
            if raw_output == "Iris Setosa":
                output[0] = 1
            elif raw_output == "Iris Versicolour":
                output[1] = 1
            else:
                output[2] = 1

            dataset.append((input, output))
    return dataset


class TestFeedforwardNeuralNetwork(TestCase):
    def test_feedforward_no_hidden_layers_squared_activation(self):
        weights = [[[2, 1], [3, 2], [4, 2]]]
        biases = [[-2, 3, 0]]
        n = FeedforwardNeuralNetwork(weights=weights,
                                     biases=biases,
                                     activation_function=lambda x: x * x,
                                     activation_function_derivative=lambda x: 2 * x,
                                     cost_function_derivative=FunctionsDerivative.mse_by_activation_derivative)
        output = n.feedforward(numpy.array([-3, 4]))

        self.assertTrue(numpy.array_equal(output, [16, 4, 16]))

    def test_feedforward_one_hidden_layer_squared_activation(self):
        weights = [[[2, 1], [3, 2], [4, 2]], [[1, 1, -1], [0, 1, 2]]]
        biases = [[-2, 3, 0], [-3, 2]]
        n = FeedforwardNeuralNetwork(weights=weights,
                                     biases=biases,
                                     activation_function=lambda x: x * x,
                                     activation_function_derivative=lambda x: 2 * x,
                                     cost_function_derivative=FunctionsDerivative.mse_by_activation_derivative)
        output = n.feedforward([-3, 4])

        self.assertTrue(numpy.array_equal(output, [1, 1444]))

    def test_feedforward_one_hidden_layer_sigmoid_activation(self):
        weights = [[[2, 1], [3, 2], [4, 2]], [[1, 1, -1], [0, 1, 2]]]
        biases = [[-2, 3, 0], [-3, 2]]
        n = FeedforwardNeuralNetwork(weights=weights,
                                     biases=biases,
                                     activation_function=Functions.sigmoid,
                                     activation_function_derivative=FunctionsDerivative.sigmoid,
                                     cost_function_derivative=FunctionsDerivative.mse_by_activation_derivative)
        output = n.feedforward([-3, 4])

        self.assertTrue(all(numpy.isclose(output, [0.10724436124, 0.94866921457])))

    def test_backpropagation(self):
        input = [8, 3]
        biases = [[1.4], [0, 0]]
        weights = [[[0.56, -0.22]], [[1.5], [-4]]]
        correct_output = [1, 0]

        n = FeedforwardNeuralNetwork(weights=weights,
                                     biases=biases,
                                     activation_function=Functions.sigmoid,
                                     activation_function_derivative=FunctionsDerivative.sigmoid,
                                     cost_function_derivative=FunctionsDerivative.mse_by_activation_derivative)
        weights_gradient_network, biases_gradient_network = n.backpropagate([(input, correct_output)], 1)


        weights_gradient = [[[-0.12128892408755970, -0.045483346532834888]],
                            [[-0.038824925911934149], [0.0045674533667444924]]]
        biases_gradient = [[-0.015161115510944963], [-0.039034865064649663, 0.0045921510903681505]]

        weights_gradient_network = [weights.tolist() for weights in weights_gradient_network]
        biases_gradient_network = [biases.flatten().tolist() for biases in biases_gradient_network]


        self.assertTrue(testAlmostEquals(biases_gradient, biases_gradient_network))
        self.assertTrue(testAlmostEquals(weights_gradient, weights_gradient_network))

    def test_backpropagation_mini_batch_size_2(self):
        input = [8, 3]
        biases = [[1.4], [0, 0]]
        weights = [[[0.56, -0.22]], [[1.5], [-4]]]
        correct_output = [1, 0]

        n = FeedforwardNeuralNetwork(weights=weights,
                                     biases=biases,
                                     activation_function=Functions.sigmoid,
                                     activation_function_derivative=FunctionsDerivative.sigmoid,
                                     cost_function_derivative=FunctionsDerivative.mse_by_activation_derivative)
        weights_gradient_network, biases_gradient_network = n.backpropagate([(input, correct_output), (input, correct_output)], 1)

        weights_gradient = [[[-0.12128892408755970, -0.045483346532834888]],
                            [[-0.038824925911934149], [0.0045674533667444924]]]
        biases_gradient = [[-0.015161115510944963], [-0.039034865064649663, 0.0045921510903681505]]

        weights_gradient_network = [weights.tolist() for weights in weights_gradient_network]
        biases_gradient_network = [biases.flatten().tolist() for biases in biases_gradient_network]

        self.assertTrue(testAlmostEquals(biases_gradient, biases_gradient_network))
        self.assertTrue(testAlmostEquals(weights_gradient, weights_gradient_network))

    def test_integration_iris(self):
        n = FeedforwardNeuralNetwork.Create(layers=[4, 4, 3],
                                            min_inclusive=-1,
                                            max_inclusive=1,
                                            activation_function=Functions.sigmoid,
                                            activation_function_derivative=FunctionsDerivative.sigmoid,
                                            cost_function_derivative=FunctionsDerivative.mse_by_activation_derivative)

        learn_dataset = read_iris_dataset("IntegrationDataset/bezdekIris.data")
        validation_dataset = read_iris_dataset("IntegrationDataset/iris.data")

        n.epoch_learn(5, 10, 0.5, [], learn_dataset, [])

        success_rate = n.get_success_rate(validation_dataset, output_function=Functions.first_max_neuron)
        self.assertGreater(success_rate, 90.0)

    def test_integration_iris_max_cpu(self):
        n = FeedforwardNeuralNetwork.Create(layers=[4, 4, 3],
                                            min_inclusive=-1,
                                            max_inclusive=1,
                                            activation_function=Functions.sigmoid,
                                            activation_function_derivative=FunctionsDerivative.sigmoid,
                                            cost_function_derivative=FunctionsDerivative.mse_by_activation_derivative)

        learn_dataset = read_iris_dataset("IntegrationDataset/bezdekIris.data")
        validation_dataset = read_iris_dataset("IntegrationDataset/iris.data")

        n.epoch_learn(5, 10, 0.5, [], learn_dataset, [], threads=multiprocessing.cpu_count())

        success_rate = n.get_success_rate(validation_dataset, output_function=Functions.first_max_neuron)
        self.assertGreater(success_rate, 90.0)

    def test_integration_iris_max_cpu_overfit(self):
        n = FeedforwardNeuralNetwork.Create(layers=[4, 100, 3],
                                            min_inclusive=-1,
                                            max_inclusive=1,
                                            activation_function=Functions.sigmoid,
                                            activation_function_derivative=FunctionsDerivative.sigmoid,
                                            cost_function_derivative=FunctionsDerivative.mse_by_activation_derivative)

        learn_dataset = read_iris_dataset("IntegrationDataset/bezdekIris.data")
        validation_dataset = read_iris_dataset("IntegrationDataset/iris.data")

        n.epoch_learn(5, 10, 0.5, [], learn_dataset, [], threads=multiprocessing.cpu_count())

        success_rate = n.get_success_rate(validation_dataset, output_function=Functions.first_max_neuron)
        self.assertGreater(success_rate, 90.0)
