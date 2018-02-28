from matplotlib import pyplot as plt
import numpy as np
from sklearn import datasets


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
        covi = np.sum([np.dot((x - mi)[:, np.newaxis], (x - mi)) for x in Xi],
                      axis=0) / Xi.shape[0]
        cov_w += pi * covi

        cov_b += pi * np.dot((mi - m)[:, np.newaxis], (mi - m))

    la, v = np.linalg.eig(
        np.dot(np.linalg.inv(cov_w), cov_b)
    )

    


if __name__ == '__main__':
    pass
