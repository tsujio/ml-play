import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn import datasets, model_selection
from neural_network import NeuralNetwork

DATA_HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

mnist = datasets.fetch_mldata('MNIST original', data_home=DATA_HOME)

x_train, x_test, y_train, y_test = model_selection.train_test_split(
    mnist.data, np.eye(10)[mnist.target.astype(int)], test_size=0.2, shuffle=True
)

NeuralNetwork.EPOCH = 1

nn = NeuralNetwork([x_train.shape[1], 100, y_train.shape[1]], rho=1e-1)

for i in range(10000):
    nn.train(x_train, y_train)

    errors = [y[nn.predict(x)] != 1
              for x, y in zip(list(x_test), list(y_test))]
    error_rate = len(x_test[errors]) / len(x_test)
    print(f"{i}: error rate={error_rate}", flush=True)
