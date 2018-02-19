from matplotlib import pyplot as plt
import numpy as np
from sklearn import datasets, model_selection


def plot_decision_boundary(data, predict):
    """ref: http://scikit-learn.org/stable/auto_examples/svm/plot_iris.html"""

    x_min, x_max = data[:, 0].min() - .5, data[:, 0].max() + .5
    y_min, y_max = data[:, 1].min() - .5, data[:, 1].max() + .5
    h = .02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class NeuralNetwork:
    EPOCH = 3000

    def __init__(self, layers, rho=1e-1):
        self.W_list = [np.random.randn(layers[i], layers[i-1] + 1)
                       for i in range(1, len(layers))]
        self.rho = rho

    def train(self, data, label):
        for i in range(self.__class__.EPOCH):
            # shuffle
            perm = np.random.permutation(len(data))
            data, label = data[perm], label[perm]

            for x, y in zip(list(data), list(label)):
                g_list = self._predict(x)

                # back propagation
                new_W_list = [np.zeros(W.shape) for W in self.W_list]
                for i in reversed(range(len(self.W_list))):
                    _g = g_list[i + 1]
                    g = g_list[i]
                    W = self.W_list[i]

                    if i == len(self.W_list) - 1:
                        # output layer
                        e = (_g - y) * _g * (1 - _g)
                    else:
                        # hidden layer
                        _W = self.W_list[i + 1][:, :-1]
                        e = np.dot(_W.T, e) * _g * (1 - _g)

                    G = np.repeat(np.concatenate([g, [1]])[np.newaxis, :],
                                  W.shape[0],
                                  axis=0)
                    E = np.repeat(e[:, np.newaxis],
                                  W.shape[1],
                                  axis=1)
                    new_W_list[i] = W - self.rho * E * G

                self.W_list = new_W_list

    def _predict(self, x):
        g_list = [x]
        for W in self.W_list:
            x = np.concatenate([g_list[-1], [1]])
            g_list.append(sigmoid(np.dot(W, x)))

        return g_list

    def predict(self, x):
        return np.argmax(self._predict(x)[-1])

    def predict_array(self, X):
        return np.array([self.predict(x) for x in list(X)])


if __name__ == '__main__':
    # prepare dataset
    moons = datasets.make_moons(n_samples=500, noise=0.1)

    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        moons[0], np.eye(2)[moons[1]], test_size=0.5, shuffle=True
    )

    # train
    nn = NeuralNetwork([2, 3, 2])
    nn.train(x_train, y_train)

    # test
    errors = [y[nn.predict(x)] != 1
              for x, y in zip(list(x_test), list(y_test))]
    error_rate = len(x_test[errors]) / len(x_test)
    print(f"error rate={error_rate}")

    # display
    plot_decision_boundary(x_test, lambda x: nn.predict_array(x))

    X_c0 = x_test[np.argmax(y_test, axis=1) == 0]
    X_c1 = x_test[np.argmax(y_test, axis=1) == 1]
    plt.scatter(X_c0[:, 0], X_c0[:, 1])
    plt.scatter(X_c1[:, 0], X_c1[:, 1])

    plt.show()
