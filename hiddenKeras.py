#Dependencies
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense



if __name__ == '__main__':
    training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # 4 x 2
    training_outputs = np.array([[0, 1, 1, 1]]).T  # 1X4 => 4 X 1

    # Neural network
    model = Sequential()
    model.add(Dense(3, input_dim=2, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='SGD', metrics=['binary_accuracy'])
    history = model.fit(training_inputs, training_outputs, epochs=1, batch_size=1)


    A = str(input("Input 1: "))
    B = str(input("Input 2: "))
    new_input = np.array([[A, B]])
    new_input = new_input.astype(float)
    print(new_input.shape)

    output = model.predict(new_input)
    print(output)