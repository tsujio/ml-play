from matplotlib import pyplot as plt
import numpy as np
from sklearn import datasets, model_selection


class MultiClassPerceptron:
    def __init__(self, x_dim, n_class, rho=1e-3):
        self.W = np.random.randn(n_class, x_dim + 1)
        self.rho = rho

    def train(self, data, label):
        while True:
            # shuffle
            perm = np.random.permutation(len(data))
            data, label = data[perm], label[perm]

            classified = True

            for x, y in zip(list(data), list(label)):
                pred = self.predict(x)
                if pred != y:
                    classified = False

                    # update weight
                    x = np.array(list(x) + [1])
                    self.W[y] = self.W[y] + self.rho * x
                    self.W[pred] = self.W[pred] - self.rho * x

            if classified:
                break

    def predict(self, x):
        x = np.array(list(x) + [1])
        return np.argmax(np.dot(self.W, x), axis=0)


if __name__ == '__main__':
    x_c0 = np.random.randn(50, 2)
    x_c1 = np.random.randn(50, 2) + np.array([0, 8])
    x_c2 = np.random.randn(50, 2) + np.array([8, 0])

    perceptron = MultiClassPerceptron(2, 3)

    perceptron.train(np.concatenate([x_c0, x_c1, x_c2]),
                     np.array([0] * 50 + [1] * 50 + [2] * 50))

    # display
    plt.scatter(x_c0[:, 0], x_c0[:, 1], label='c0')
    plt.scatter(x_c1[:, 0], x_c1[:, 1], label='c1')
    plt.scatter(x_c2[:, 0], x_c2[:, 1], label='c2')

    W = perceptron.W
    w_0_1 = W[0] - W[1]
    w_1_2 = W[1] - W[2]
    w_2_0 = W[2] - W[0]
    y_0_1 = lambda x: w_0_1[0] / -w_0_1[1] * x + w_0_1[2] / -w_0_1[1]
    y_1_2 = lambda x: w_1_2[0] / -w_1_2[1] * x + w_1_2[2] / -w_1_2[1]
    y_2_0 = lambda x: w_2_0[0] / -w_2_0[1] * x + w_2_0[2] / -w_2_0[1]
    x = np.arange(-2, 10, 0.1)
    plt.plot(x, y_0_1(x), label='c0 - c1')
    plt.plot(x, y_1_2(x), label='c1 - c2')
    plt.plot(x, y_2_0(x), label='c2 - c0')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
