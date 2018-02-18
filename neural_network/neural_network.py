from matplotlib import pyplot as plt
import numpy as np
from sklearn import datasets, model_selection


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class NeuralNetwork:
    pass

if __name__ == '__main__':
    # prepare dataset
    X_c0 = 5 * np.random.randn(100, 2)
    X_c1 = 5 * np.random.randn(100, 2) + np.array([10, 10])

    data = np.concatenate([X_c0, X_c1])
    label = np.array([[1, 0] for i in range(100)] +
                     [[0, 1] for i in range(100)])

    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        data, label, test_size=0.2, shuffle=True
    )

    # train
    nn = WidrowHoff(n_dim=2, n_class=2)
    nn.train(data, label)

    # test
    errors = [y[nn.predict(x)] != 1
              for x, y in zip(list(x_test), list(y_test))]
    error_rate = len(x_test[errors]) / len(x_test)
    print(f"error rate={error_rate}")
