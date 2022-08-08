import numpy as np


class NeuralNetwork:
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def sigmoid_derivative(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))


if __name__ == '__main__':
    neural_network = NeuralNetwork()

    training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # 4 X 2
    training_outputs = np.array([[0, 1, 1, 1]]).T  # 1 X 4 => 4 X 1
    # print(training_outputs.shape)

    bh = 0.5
    bo = 0.5
    wh = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    wo = np.array([[0.7], [0.8], [0.9]])
    learningrate = 0.05
    for epoch in range(10000):
        # Feed Forward
        inh = np.dot(training_inputs, wh) + bh

        outh = neural_network.sigmoid(inh)

        ino = np.dot(outh, wo) + bo

        outo = neural_network.sigmoid(ino)

        error = training_outputs - outo

        cost = 0.5 * np.sum(np.square(error))

        # print(cost)
        # Phase 1 Derivatives
        derror_douto = outo - training_outputs
        douto_dino = neural_network.sigmoid_derivative(ino)
        dino_dwo = outh

        derror_dino = derror_douto * douto_dino    # bias output layer
        derror_dwo = np.dot(dino_dwo.T, derror_dino)

        # Phase 2 use derror_doutput and doutput_dzout
        dino_douth = wo
        derror_douth = np.dot(derror_dino, dino_douth.T)
        douth_dinh = neural_network.sigmoid_derivative(inh)
        dinh_dwh = training_inputs

        # Phase 2 bias derivative
        derror_dinh = (douth_dinh * derror_douth)
        derror_dwh = np.dot(dinh_dwh.T, derror_dinh)

        wo -= learningrate*derror_dwo
        wh -= learningrate*derror_dwh

        for i in derror_dino:
            bo -= learningrate*i

        for i in derror_dinh:
            bh -= learningrate*i


print(wo, bo)

A = str(input("Input 1 :"))
B = str(input("Input 2 :"))

new_input = np.array([A, B])
new_input = new_input.astype(float)

z = np.dot(new_input, wh) + bh
hidden_output = neural_network.sigmoid(z)
expZout = np.dot(hidden_output, wo) + bo
expout = neural_network.sigmoid(expZout)


print(expout)
