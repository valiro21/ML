import random
import time
from concurrent import futures
from copy import deepcopy

import functools
import numpy

from ML.Common import argcheck
from ML.Core import Functions


class FeedforwardNeuralNetwork(object):
    def __init__(self, weights, biases, activation_function,
                 activation_function_derivative, cost_function_derivative):
        """
        Initialize a new instance of :class:`FeedforwardNeuralNetwork`.
        :param weights: The initial iterable weights of the network.
        :param biases: The initial iterable biases of the network.
        :param activation_function: The (non-None) activation function to use for all layers.
        :param activation_function_derivative: The (non-None) activation function derivative.
        :param cost_function_derivative: The (non-None) cost function derivative.
        """

        argcheck.throw_on_non_iterable(weights, "weights")
        argcheck.throw_on_empty(weights, "weights")
        argcheck.throw_on_non_iterable(biases, "biases")
        argcheck.throw_on_empty(biases, "weights")
        argcheck.throw_if_false(len(weights) == len(biases),
                                "The number of layers in the weights and biases must be the same")
        argcheck.throw_on_none(activation_function, "activation_function")
        argcheck.throw_on_none(activation_function_derivative, "activation_function_derivative")
        argcheck.throw_on_none(cost_function_derivative, "cost_function_derivative")

        self.layers = []
        self.weights = []
        self.biases = []
        for layer in zip(weights, biases):
            layer_weights = numpy.array(layer[0])
            layer_biases = numpy.array(layer[1])
            argcheck.throw_if_false(layer_weights.shape[0] == layer_biases.shape[0],
                                    "The layer size must be the same in for the weights and biases")

            self.weights.append(layer_weights)
            self.biases.append(layer_biases)
            self.layers.append(layer_weights.size)

        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative
        self.cost_function_derivative = cost_function_derivative
        self.dws = deepcopy(self.weights)
        self.dbs = deepcopy(self.biases)

    @staticmethod
    def Create(layers, min_inclusive, max_inclusive, activation_function,
               activation_function_derivative, cost_function_derivative):
        """
        Create a valid :class:`FeedforwardNeuralNetwork` with random weights and biases with values between [min, max]
        :param layers: A iterable list with the number of neurons in each layer for a minimum of two layers.
        :param min_inclusive: The (non-nan, non-inf) minimum inclusive value for the weights and biases
        :param max_inclusive: The (non-nan, non-inf) maximum inclusive value for the weights and biases
        :param activation_function: The (non-None) activation function to use for all layers.
        :param activation_function_derivative: The (non-None) activation function derivative.
        :param cost_function_derivative: The (non-None) cost function derivative.
        :return:
        """
        argcheck.throw_on_non_iterable(layers, "layers")
        argcheck.throw_if_false(len(layers) > 1, "There must be at least an input and an output layer.")
        argcheck.throw_on_nan_or_inf(min_inclusive, "min_inclusive")
        argcheck.throw_on_nan_or_inf(max_inclusive, "max_inclusive")
        argcheck.throw_on_none(activation_function, "activation_function")
        argcheck.throw_on_none(activation_function_derivative, "activation_function_derivative")
        argcheck.throw_on_none(cost_function_derivative, "cost_function_derivative")

        weights = []
        biases = []

        random.seed(4533)
        for l in range(0, len(layers) - 1):
            biases.append([random.uniform(min_inclusive, max_inclusive) for i in range(layers[l + 1])])
            weights.append([[random.uniform(min_inclusive, max_inclusive) for j in range(layers[l])] for i in range(layers[l + 1])])
        return FeedforwardNeuralNetwork(weights=weights,
                                        biases=biases,
                                        activation_function=activation_function,
                                        activation_function_derivative=activation_function_derivative,
                                        cost_function_derivative=cost_function_derivative)

    def _feedforward(self, input, return_intermediary_results=False):
        """
        Given the input layer, compute the values of the last (output) layer.
        :param input: The iterable fist input layer.
        :return: The iterable values for the last (output) layer.
        """
        argcheck.throw_on_non_iterable(input, "input")

        activation = numpy.array(input)
        activations = []
        Z = []
        for biases, weights in zip(self.biases, self.weights):
            z = numpy.dot(weights, activation) + biases
            if return_intermediary_results:
                activations.append(activation)
                Z.append(z)
            activation = self.activation_function(z)
        if return_intermediary_results:
            return (activation, Z, activations)
        return activation

    def feedforward(self, input):
        """
        Given the input layer, compute the values of the last (output) layer.
        :param input: The iterable fist input layer.
        :return: The iterable values for the last (output) layer.
        """
        argcheck.throw_on_non_iterable(input, "input")
        return self._feedforward(input)

    def _log_training_data_callback(self, training_validation_function, learning_rate, output, correct_output, Z, activations):
        if training_validation_function is not None:
            f_correct_output = training_validation_function(correct_output)
            if Functions.all(numpy.isclose(training_validation_function(output), f_correct_output)):
                self.training_passed_count += 1
            
            # Predict the next output by the derivative and learning rate
            if learning_rate is not None:
                dz = self.cost_function_derivative(output, correct_output, Z[-1], self.activation_function_derivative)
                next_output = output - self.activation_function_derivative(Z[-1]) * dz * learning_rate
                if Functions.all(numpy.isclose(training_validation_function(next_output), f_correct_output)):
                    self.predicted_next_passed_count += 1
                else:
                    if Functions.all(numpy.isclose(training_validation_function(output), f_correct_output)):
                        print("Prediction mismatch: ", output, next_output)
                        print("Network result:", training_validation_function(output))
                        print("Next result:", training_validation_function(next_output))
                        print("Correct result:", f_correct_output)

    def _single_test_backpropagate(self, input, correct_output, feedforward_callback=None):
        """
        Compute the gradient for a single input / output pair and store it internally.
        :param input: The input for the network.
        :param correct_output: The output that the network is supposed to compute.
        :param feedforward_callback: An optional function to call (output, correct_output, z values, activations) after the feedforward and before the actual gradient computation.
        """
        output, Z, activations = self._feedforward(input, return_intermediary_results=True)
        dz = self.cost_function_derivative(output, correct_output, Z[-1], self.activation_function_derivative)

        if feedforward_callback is not None:
            feedforward_callback(output, correct_output, Z, activations)

        for i in range(len(self.weights)):
            weights_gradient = numpy.multiply(numpy.transpose([dz]), activations[-i - 1])
            self.dws[-i - 1] = numpy.add(self.dws[-i - 1], weights_gradient)
            self.dbs[-i - 1] = numpy.add(self.dbs[-i - 1], dz)
            if i < len(self.weights) - 1:
                z = Z[-i-2]
                dz = numpy.dot(dz, numpy.dot(self.weights[-i - 1], self.activation_function_derivative(z)))

    def backpropagate(self, mini_batch_dataset, threads=1, feedforward_callback=None):
        """
        Compute the gradient for the given sample of the training dataset.

        This is not thread-safe.
        :param mini_batch_dataset: The iterable sample of the training dataset.
        :param threads: The positive number of threads to use for the computation.
        :param feedforward_callback: An optional function to call (output, correct_output, z values, activations) after the feedforward is completed for a testcase and before the actual gradient computation.
        :return: A tuple of the normalized gradients for the weights and biases.
        """
        argcheck.throw_on_non_iterable(mini_batch_dataset, "mini_batch_dataset")
        argcheck.throw_on_non_positive(threads, "threads")

        mini_batch_size = len(mini_batch_dataset)
        for i in range(0, len(self.dws)):
            self.dws[i] = numpy.multiply(self.dws[i], 0)
            self.dbs[i] = numpy.multiply(self.dbs[i], 0)

        with futures.ThreadPoolExecutor(max_workers=threads) as executor:
            threads = [executor.submit(self._single_test_backpropagate, data[0], data[1], feedforward_callback)
                        for data in mini_batch_dataset]
            futures.wait(threads, return_when=futures.FIRST_EXCEPTION)

        self.dws = numpy.multiply(self.dws, 1.0 / mini_batch_size)
        self.dbs = numpy.multiply(self.dbs, 1.0 / mini_batch_size)

        return self.dws, self.dbs

    def learn(self, mini_batch_dataset, learning_rate, threads=1, feedforward_callback=None):
        """
        Change the weights and biases according to the soon to be computed gradient and the learning_rate.
        :param mini_batch_dataset: The iterable sample of the training dataset.
        :param learning_rate: The positive learning rate to use for improving.
        :param threads: The positive number of threads to use for the computation.
        :param feedforward_callback: An optional function to call (output, correct_output, z values, activations) with the output after the feedforward is completed for a testcase and before the actual gradient computation.
        """
        argcheck.throw_on_non_iterable(mini_batch_dataset, "mini_batch_dataset")
        argcheck.throw_on_non_positive(learning_rate, "learning_rate")
        argcheck.throw_on_non_positive(threads, "threads")

        weights_gradient, biases_gradient = self.backpropagate(mini_batch_dataset, threads=threads, feedforward_callback=feedforward_callback)
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - learning_rate * weights_gradient[i]
            self.biases[i] = self.biases[i] - learning_rate * biases_gradient[i]

    def epoch_learn(self, number_of_epoch, mini_batch_size, starting_learning_rate, learning_rate_changes, 
            training_dataset, validation_dataset, output_function=Functions.no_change, threads=1,
            log_training_rate=False, predict_next_training_rate=False):
        """

        :param nrepoch: The positive number of epoch to train for.
        :param mini_batch_size: The positive number of sample size to use for one training session.
        :param starting_learning_rate: The positive initial learning rate.
        :param learning_rate_changes: The iterable changes in learning rate after an epoch.
        :param training_dataset: The iterable dataset to use for training.
        :param validation_dataset: The iterable dataset to validate the network against.
        :param output_function: The (non-None) output function for the output layer used when validating.
        :param threads: The positive number of threads to use for the computation.
        :param log_training_rate: A value indicating whether the success rate for the training dataset should be logged.
        :param predicted_next_training_rate: A valud indicating whether the network should predict the next success rate for the training dataset.
        """
        argcheck.throw_on_non_positive(number_of_epoch, "number_of_epoch")
        argcheck.throw_on_non_positive(mini_batch_size, "mini_batch_size")
        argcheck.throw_on_non_positive(starting_learning_rate, "starting_learning_rate")
        argcheck.throw_on_non_iterable(learning_rate_changes, "learning_rate_changes")
        argcheck.throw_on_non_iterable(training_dataset, "training_dataset")
        argcheck.throw_on_non_iterable(validation_dataset, "validation_dataset")
        argcheck.throw_on_none(output_function, "output_function")
        argcheck.throw_on_non_positive(threads, "threads")

        if predict_next_training_rate:
            argcheck.throw_if_false(log_training_rate, "Can't predict the next training rate without computing the actual training rate.")

        if len(validation_dataset) == 0:
            argcheck.throw_if_true(len(learning_rate_changes) > 0,
                                   "If there is no validation data then the learning_rate_changes must also be empty")

        use_validation_data = len(validation_dataset) != 0
        if use_validation_data:
            passed_count = self.get_passed_count(validation_dataset, output_function=output_function)
            print("Starting validation rate: {0} / {1}".format(passed_count, len(validation_dataset)))

        for epoch in range(number_of_epoch):
            # Change the learning rate dynamically by the validation rate
            learning_rate = starting_learning_rate

            self.epoch = epoch
            if use_validation_data:
                current_rate = passed_count / len(validation_dataset)
                for next_validation_rate, next_learning_rate in learning_rate_changes:
                    if next_validation_rate <= current_rate:
                        learning_rate = next_learning_rate

            start = time.time()
            random.shuffle(training_dataset)
           
            # Set training validation callback
            callback = None
            if log_training_rate:
                callback = functools.partial(self._log_training_data_callback, output_function, None)
                self.training_passed_count = 0
                if predict_next_training_rate:
                    self.predicted_next_passed_count = 0
                    callback = functools.partial(self._log_training_data_callback, output_function, learning_rate)

            chunks = [training_dataset[i:i + mini_batch_size] for i in range(0, len(training_dataset), mini_batch_size)]
            for chunk in chunks:
                self.learn(chunk, learning_rate, threads=threads, feedforward_callback=callback)

            end = time.time()

            if use_validation_data or log_training_rate:
                print("Epoch #{0}:".format(epoch + 1))

                if use_validation_data:
                    passed_count = self.get_passed_count(validation_dataset,
                                                         output_function)
                    print("Learning rate used: {0}".format(learning_rate))
                    print(("Time taken: {0} seconds\n" +
                           "Validation rate: {1} / {2}, {3}%")
                          .format(end - start, passed_count,
                                  len(validation_dataset),
                                  passed_count / len(validation_dataset) * 100.0))
                if log_training_rate:
                    print("Training set validation rate: {0} / {1}, {2}%"
                          .format(self.training_passed_count,
                                  len(training_dataset),
                                  self.training_passed_count / len(training_dataset) * 100.0))
                    if predict_next_training_rate:
                        print("Next predicted training set validation rate: {0} / {1}, {2}%"
                              .format(self.predicted_next_passed_count,
                                      len(training_dataset),
                                      self.predicted_next_passed_count / len(training_dataset) * 100.0))
                print("")

    def get_passed_count(self, validation_dataset,
                         output_function=Functions.no_change):
        """
        Compute how many inputs does the neural network recognize from the vaidation_dataset.
        :param validation_dataset: The iterable dataset to validate the network against.
        :param output_function: The (non-None) output function for the output layer used when validating.
        :return:
        """
        argcheck.throw_on_non_iterable(validation_dataset, "validation_dataset")
        argcheck.throw_on_none(output_function, "output_function")

        passed = 0
        for input, output in validation_dataset:
            network_output = self.feedforward(input)
            if Functions.all(numpy.isclose(output_function(network_output.tolist()), output_function(output))):
                passed += 1

        return passed

    def get_success_rate(self, validation_dataset, output_function=Functions.no_change):
        """
        Compute how many inputs in percentage does the neural network recognize from the vaidation_dataset.
        :param validation_dataset: The iterable dataset to validate the network against.
        :param output_function: The (non-None) output function for the output layer used when validating.
        :return:
        """
        argcheck.throw_on_non_iterable(validation_dataset, "validation_dataset")
        argcheck.throw_on_none(output_function, "output_function")

        passed = self.get_passed_count(validation_dataset, output_function=output_function)

        return passed / len(validation_dataset) * 100.0
