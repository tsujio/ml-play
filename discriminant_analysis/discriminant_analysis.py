from matplotlib import pyplot as plt
import numpy as np


def discriminant_analysis(X, y):
    classes = np.unique(y)
    X0, X1 = X[y == classes[0]], X[y == classes[1]]
    m0, m1 = X0.mean(axis=0), X1.mean(axis=0)
    Sw = np.sum(
        [np.sum([np.dot((x - mi)[:, np.newaxis],
                        (x - mi)[np.newaxis, :]) for x in Xi],
                axis=0)
         for Xi, mi in [(X0, m0), (X1, m1)]],
        axis=0)
    Sb = (len(X0) * len(X1)) / len(X) * np.dot(
        (m0 - m1)[:, np.newaxis],
        (m0 - m1)[np.newaxis, :]
    )

    la, v = np.linalg.eig(
        np.dot(np.linalg.inv(Sw), Sb)
    )

    w = v.T[np.argmax(la)]
    m = (m0 + m1) / 2
    return lambda x: -w[0]/w[1] * x + w[0]/w[1] * m[0] + m[1]


if __name__ == '__main__':
    n, dim = 50, 2
    C = np.array([[0., -0.23], [0.83, .23]])
    X = np.r_[np.dot(np.random.randn(n, dim), C),
              np.dot(np.random.randn(n, dim), C) + np.array([1, 1])]
    y = np.hstack((np.zeros(n), np.ones(n)))

    plt.scatter(X[:, 0], X[:, 1], c=y)
    x = np.arange(-2, 2, 0.1)
    plt.plot(x, discriminant_analysis(X, y)(x))
    plt.show()
