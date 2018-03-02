from matplotlib import pyplot as plt
import numpy as np


def discriminant_analysis(X, y):
    # between-class covariance matrix
    cov_b = np.zeros((X.shape[1], X.shape[1]))
    # within-class covariance matrix
    cov_w = np.zeros((X.shape[1], X.shape[1]))

    m = np.array([np.sum(X[:, i]) / X.shape[0]
                  for i in range(X.shape[1])])
    for yi in np.unique(y):
        Xi = X[y == yi]
        pi = len(y[y == yi]) / len(y)
        mi = np.array([np.sum(Xi[:, i]) / Xi.shape[0]
                       for i in range(Xi.shape[1])])
#        covi = np.sum([np.dot((x - mi)[:, np.newaxis],
#                              (x - mi)[np.newaxis, :]) for x in Xi],
#                      axis=0) / Xi.shape[0]
        covi = np.sum([
            (x - mi).reshape(Xi.shape[1], 1) * (x - mi).reshape(1, Xi.shape[1])
            for x in Xi
        ], axis=0) / Xi.shape[0]
        cov_w += pi * covi

#        cov_b += pi * np.dot((mi - m)[:, np.newaxis], (mi - m)[np.newaxis, :])
        cov_b += pi * (mi - m).reshape(Xi.shape[1], 1) * (mi - m).reshape(1, Xi.shape[1])

    la, v = np.linalg.eig(
#        np.dot(np.linalg.inv(cov_w), cov_b)
        np.linalg.inv(cov_w) * cov_b
    )
    return v[np.argmax(la)]


if __name__ == '__main__':
    n, dim = 300, 2
    C = np.array([[0., -0.23], [0.83, .23]])
    X = np.r_[np.dot(np.random.randn(n, dim), C),
              np.dot(np.random.randn(n, dim), C) + np.array([1, 1])]
    y = np.hstack((np.zeros(n), np.ones(n)))

    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()
