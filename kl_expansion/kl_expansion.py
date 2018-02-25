from matplotlib import pyplot as plt
import numpy as np
from sklearn import datasets, model_selection


def kl_expansion(d, X):
    m = np.array([np.sum(X[:, i]) / X.shape[0]
                  for i in range(X.shape[1])])
    cov = np.sum([(x - m) * (x - m)[:, np.newaxis] for x in X],
                 axis=0) / X.shape[0]
    la, v = np.linalg.eig(cov)
    indices = np.array(list(reversed(np.argsort(la))))
    return v[indices[:d]]


if __name__ == '__main__':
    X = np.array([3, 1]) * np.random.randn(50, 2)
    w = np.pi / 3
    A = np.array([[np.cos(w), -np.sin(w)],
                  [np.sin(w), np.cos(w)]])
    X = np.array([np.dot(A, x) for x in X])

    B = kl_expansion(1, X)

    from sklearn import decomposition
    pca = decomposition.PCA(n_components=1)
    Xt = pca.fit(X)
    print(B)
    print(pca.components_)

    plt.scatter(X[:, 0], X[:, 1])
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
#    plt.show()
