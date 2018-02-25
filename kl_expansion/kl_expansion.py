from matplotlib import pyplot as plt
import numpy as np
from sklearn import datasets


def kl_expansion(d, X):
    m = np.array([np.sum(X[:, i]) / X.shape[0]
                  for i in range(X.shape[1])])
    cov = np.sum([(x - m) * (x - m)[:, np.newaxis] for x in X],
                 axis=0) / X.shape[0]
    la, v = np.linalg.eig(cov)
    indices = np.array(list(reversed(np.argsort(la))))
    return v.T[indices[:d]]


if __name__ == '__main__':
    iris = datasets.load_iris()

    A = kl_expansion(2, iris.data)
    from sklearn import decomposition
    pca = decomposition.PCA(n_components = 2)
    pca.fit(iris.data)
#    A = pca.components_

    X = np.array([np.dot(A, x) for x in iris.data])

    plt.scatter(X[:, 0], X[:, 1], c=iris.target)
    plt.show()
