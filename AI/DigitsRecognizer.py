from ML.Core import FeedforwardNeuralNetwork, Functions, FunctionsDerivative
import os
from ML.Common import argcheck

def read_input(input_reader):
    input = input_reader.read(28 * 28)
    return list(map(lambda x: float(x) / 255.0, input))


def read_output(output_reader):
    output = [0 for i in range(10)]
    digit = output_reader.read(1)[0]
    output[digit] = 1

    return output


def read_dataset(input_file, output_file):
    with open(input_file, 'rb') as finput, open(output_file, 'rb') as foutput:
        input_number = int.from_bytes(finput.read(4), byteorder='big')
        assert int(input_number) == 2051

        output_number = int.from_bytes(foutput.read(4), byteorder='big')
        assert int(output_number) == 2049

        number_of_items = int.from_bytes(finput.read(4), byteorder='big')
        number_of_items_output = int.from_bytes(foutput.read(4), byteorder='big')

        assert number_of_items == number_of_items_output, "The should be as many inputs as there are outputs"

        width = int.from_bytes(finput.read(4), byteorder='big')
        height = int.from_bytes(finput.read(4), byteorder='big')

        assert width == 28 and height == 28, "Input must be an image of size 28 x 28"

        dataset = []
        for i in range(number_of_items):
            input = read_input(finput)
            output = read_output(foutput)

            dataset.append((input, output))

    return dataset


def run(threads=1):
    argcheck.throw_on_non_positive(threads, "threads")    

    file_path = os.path.dirname(os.path.realpath(__file__))
    learn_dataset = read_dataset(os.path.join(file_path, "./datasets/digits-dataset/train-images-idx3-ubyte"),
                                 os.path.join(file_path,"./datasets/digits-dataset/train-labels-idx1-ubyte"))
    validation_dataset = read_dataset(os.path.join(file_path,"./datasets/digits-dataset/t10k-images-idx3-ubyte"),
                                      os.path.join(file_path,"./datasets/digits-dataset/t10k-labels-idx1-ubyte"))

    print("Finished reading dataset!")

    n = FeedforwardNeuralNetwork.Create(layers=[28 * 28, 100, 10],
                                        min_inclusive=-0.5,
                                        max_inclusive=0.5,
                                        activation_function=Functions.sigmoid,
                                        activation_function_derivative=FunctionsDerivative.sigmoid,
                                        cost_function_derivative=FunctionsDerivative.mse_by_activation_derivative)

    n.epoch_learn(30, 10, 0.01, [], learn_dataset, validation_dataset, output_function=Functions.first_max_neuron, threads=threads)

if __name__ == '__main__':
    run(threads=2)
