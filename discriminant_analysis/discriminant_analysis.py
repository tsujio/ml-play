from matplotlib import pyplot as plt
import numpy as np


def _discriminant_analysis(X, y):
    # between-class covariance matrix
    cov_b = np.zeros((X.shape[1], X.shape[1]))
    # within-class covariance matrix
    cov_w = np.zeros((X.shape[1], X.shape[1]))

    m = np.array([np.sum(X[:, i]) / X.shape[0]
                  for i in range(X.shape[1])])
    M = np.zeros(X.shape[1])
    for yi in np.unique(y):
        Xi = X[y == yi]
        pi = len(y[y == yi]) / len(y)
        mi = np.array([np.sum(Xi[:, i]) / Xi.shape[0]
                       for i in range(Xi.shape[1])])
        covi = np.sum([np.dot((x - mi)[:, np.newaxis],
                              (x - mi)[np.newaxis, :]) for x in Xi],
                      axis=0) / Xi.shape[0]
#        covi = np.sum([
#            (x - mi).reshape(Xi.shape[1], 1) * (x - mi).reshape(1, Xi.shape[1])
#            for x in Xi
#        ], axis=0) / Xi.shape[0]
#        print(covi)
        cov_w += pi * covi

        cov_b += pi * np.dot((mi - m)[:, np.newaxis], (mi - m)[np.newaxis, :])
#        cov_b += pi * (mi - m).reshape(Xi.shape[1], 1) * (mi - m).reshape(1, Xi.shape[1])

        M += mi / len(np.unique(y))

    la, v = np.linalg.eig(
        np.dot(np.linalg.inv(cov_w), cov_b)
#        np.linalg.inv(cov_w) * cov_b
    )

    w = v.T[np.argmax(la)]
    return lambda x: -w[0]/w[1] * x + -w[0]/w[1] * M[0] + M[1]#v[np.argmax(la)]


def discriminant_analysis(X, y):
    X0, X1 = X[y == 0], X[y == 1]
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

    for yi in np.unique(y):
        Xi = X[y == yi]
        mi = Xi.mean(axis=0)

        Sw += np.sum([np.dot((x - mi)[:, np.newaxis],
                             (x - mi)[np.newaxis, :]) for x in Xi],
                     axis=0)
#    Sb = () * 

    # between-class covariance matrix
    cov_b = np.zeros((X.shape[1], X.shape[1]))
    # within-class covariance matrix
    cov_w = np.zeros((X.shape[1], X.shape[1]))

    m = np.array([np.sum(X[:, i]) / X.shape[0]
                  for i in range(X.shape[1])])
    M = np.zeros(X.shape[1])
    for yi in np.unique(y):
        Xi = X[y == yi]
        pi = len(y[y == yi]) / len(y)
        mi = np.array([np.sum(Xi[:, i]) / Xi.shape[0]
                       for i in range(Xi.shape[1])])
        covi = np.sum([np.dot((x - mi)[:, np.newaxis],
                              (x - mi)[np.newaxis, :]) for x in Xi],
                      axis=0) / Xi.shape[0]
#        covi = np.sum([
#            (x - mi).reshape(Xi.shape[1], 1) * (x - mi).reshape(1, Xi.shape[1])
#            for x in Xi
#        ], axis=0) / Xi.shape[0]
#        print(covi)
        cov_w += pi * covi

        cov_b += pi * np.dot((mi - m)[:, np.newaxis], (mi - m)[np.newaxis, :])
#        cov_b += pi * (mi - m).reshape(Xi.shape[1], 1) * (mi - m).reshape(1, Xi.shape[1])

        M += mi / len(np.unique(y))

    la, v = np.linalg.eig(
        np.dot(np.linalg.inv(cov_w), cov_b)
#        np.linalg.inv(cov_w) * cov_b
    )

    w = v.T[np.argmax(la)]
    return lambda x: -w[0]/w[1] * x + -w[0]/w[1] * M[0] + M[1]#v[np.argmax(la)]



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
