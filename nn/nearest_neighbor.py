import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn import datasets, model_selection

DATA_HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

K = int(os.environ.get('NN_K', 1))
CLUSTER_SIZE = int(os.environ.get('NN_CLUSTER_SIZE', 1))


def distance(x, y):
    return np.sqrt(np.sum((x - y)**2))


def predict(x, prototypes):
    # calculate distances from x for each prototype
    distances = []
    for cls, data in prototypes.items():
        distances.extend([(int(cls), distance(x, y)) for y in data])

    distances.sort(key=lambda d: d[1])

    # count classes of top K
    class_count = [0] * 10
    for cls, _ in distances[:K]:
        class_count[cls] += 1

    # return the class having maximum class count
    return max(enumerate(class_count), key=lambda i: i[1])[0]


if __name__ == '__main__':
    mnist = datasets.fetch_mldata('MNIST original', data_home=DATA_HOME)

    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        mnist.data, mnist.target, test_size=0.5, shuffle=True
    )

    # normalize
    x_train = x_train / 255
    x_test = x_test / 255

    # choose prototypes for each class
    prototypes = {}
    for i in range(10):
        data = x_train[y_train == i]
        prototypes[str(i)] = data[:CLUSTER_SIZE]

    # predict
    predicts = np.array([predict(x, prototypes) for x in x_test])
    accuracy = len(x_test[predicts == y_test]) / len(x_test)

    print(f"accuracy = {accuracy}")
