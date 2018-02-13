import os
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from sklearn import datasets, model_selection


class Perceptron:
    def __init__(self, x_dim, rho=1e-3):
        self.w = np.random.randn(x_dim + 1)
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
                    self.w = self.w - pred * self.rho * x

            if classified:
                break

    def predict(self, x):
        x = np.array(list(x) + [1])
        return 1 if np.dot(self.w, x) > 0 else -1


if __name__ == '__main__':
    dataset = datasets.load_iris()

    x_train, y_train = dataset.data, dataset.target

    perceptron = Perceptron(x_dim=2)

    # preprocess train data
    mask = np.bitwise_or(y_train == 0, y_train == 1)
    x_train = x_train[mask][:, 2:]
    y_train = y_train[mask]
    y_train = np.array([-1 if y == 0 else 1 for y in y_train])

    # train
    perceptron.train(x_train, y_train)

    # display
    fp = FontProperties(fname=r'C:\Windows\Fonts\meiryo.ttc', size=12)

    x_c0 = x_train[y_train == -1]
    x_c1 = x_train[y_train == 1]
    plt.scatter(x_c0[:, 0], x_c0[:, 1], label='Setosa')
    plt.scatter(x_c1[:, 0], x_c1[:, 1], label='Versicolour')

    w = perceptron.w
    y = lambda x: w[0] / -w[1] * x + w[2] / -w[1]
    x = np.arange(1, 5, 0.1)
    plt.plot(x, y(x))

    plt.xlabel('花弁の長さ', fontproperties=fp)
    plt.ylabel('花弁の幅', fontproperties=fp)
    plt.title('パーセプトロンによるアヤメ科の花の分類', fontproperties=fp)

    plt.legend()
    plt.show()
