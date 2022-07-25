import numpy as np


class NeuralNetwork():
    # def __init__(self):
    #     np.random.seed(1)

    #     self.synaptic_weights = 2*np.random.random((2, 1))-1

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def sigmoid_derivative(self, x):
        """
        The derivative of the sigmoid function used to
        calculate necessary weight adjustments
        """
        return x * (1 - x)


if __name__ == '__main__':
    neural_network = NeuralNetwork()

    training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # 4 x 2
    training_outputs = np.array([[0, 1, 1, 1]]).T  # 1X4 => 4 X 1

    bias = np.array([.5])
    learningRate = np.array([.05])

    weights = np.array([[.2], [.3]])  # 2 x 1

    for epoch in range(10000):

        z = np.dot(training_inputs, weights) + bias  # z   4 X 1
        output = neural_network.sigmoid(z)

        # print(output)

        error = training_outputs - output

        # print(error)

        cost = .5*np.sum(np.square(error))

        derror_doutput = output - training_outputs
        doutput_dz = neural_network.sigmoid_derivative(output)
        dz_dweights = training_inputs.T

        derror_dz = doutput_dz * derror_doutput

        derror_dweights = np.dot(dz_dweights, derror_dz)

        weights -= learningRate*derror_dweights

        for i in derror_dz:
            bias -= learningRate*i

        # print(derror_doutput)
        # print(doutput_dz)
        # print(dz_dweights)
        # print(derror_dz)
        # print(derror_dweights)
    print(weights)
    A = str(input("Input 1: "))
    B = str(input("Input 2: "))
    new_input = np.array([A, B])
    new_input = new_input.astype(float)
    z = np.dot(new_input, weights) + bias  # z   4 X 1
    output = neural_network.sigmoid(z)
    print(output)
