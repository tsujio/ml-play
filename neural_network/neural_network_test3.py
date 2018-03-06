from matplotlib import pyplot as plt
import numpy as np
from sklearn import datasets, model_selection
from neural_network import NeuralNetwork, plot_decision_boundary


moons = datasets.make_moons(n_samples=500, noise=0.1)

x_train, x_test, y_train, y_test = model_selection.train_test_split(
    moons[0], np.eye(2)[moons[1]], test_size=0.5, shuffle=True
)

NeuralNetwork.EPOCH = 1

nn = NeuralNetwork([2, 3, 2], rho=1e-1)

error_rate_history = []
for i in range(500):
    nn.train(x_train, y_train)

    errors = [y[nn.predict(x)] != 1
              for x, y in zip(list(x_test), list(y_test))]
    error_rate = len(x_test[errors]) / len(x_test)
    print(f"{i}: error rate={error_rate}")
    error_rate_history.append(error_rate)

plot_decision_boundary(x_test, lambda x: nn.predict_array(x))
plt.scatter(x_test[:, 0], x_test[:, 1], c=np.argmax(y_test, axis=1))
plt.show()

plt.plot(list(range(500)), error_rate_history)
plt.xlabel('epoch')
plt.ylabel('error rate')
plt.show()
