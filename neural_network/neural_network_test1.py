from matplotlib import pyplot as plt
import numpy as np
from sklearn import datasets, model_selection
from neural_network import NeuralNetwork, plot_decision_boundary


data, target = datasets.make_classification(n_samples=1000,
                                            n_features=2,
                                            n_redundant=0,
                                            n_informative=2,
                                            n_clusters_per_class=1,
                                            n_classes=3)

x_train, x_test, y_train, y_test = model_selection.train_test_split(
    data, np.eye(3)[target], test_size=0.2, shuffle=True
)

NeuralNetwork.EPOCH = 1

nn = NeuralNetwork([2, 3, 3], rho=1e-1)

for i in range(1000):
    nn.train(x_train, y_train)

    errors = [y[nn.predict(x)] != 1
              for x, y in zip(list(x_test), list(y_test))]
    error_rate = len(x_test[errors]) / len(x_test)
    print(f"{i}: error rate={error_rate}")

plot_decision_boundary(x_test, lambda x: nn.predict_array(x))
plt.scatter(x_test[:, 0], x_test[:, 1], c=np.argmax(y_test, axis=1))
plt.show()
