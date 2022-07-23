import numpy as np


class NeuralNetwork():
    def __init__(self):
        np.random.seed(1)

        self.synaptic_weights = 2*np.random.random((2, 1))-1

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))


if __name__ == '__main__':
    neural_network = NeuralNetwork()

    training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # 4 x 2
    training_outputs = np.array([[0, 1, 1, 1]]).T  # 1X4 => 4 X 1

    bias = np.array([.5])

    weights = np.array([[.4], [.3]])  # 2 x 1

    z = np.dot(training_inputs, weights) + bias  # z   4 X 1
    output = neural_network.sigmoid(z)

    print(output)

    error = training_outputs - output

    print(error)

    cost = .5*np.sum(np.square(error))

    print(cost)
