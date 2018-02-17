import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn import datasets, model_selection
from widrow_hoff import WidrowHoff

DATA_HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

mnist = datasets.fetch_mldata('MNIST original', data_home=DATA_HOME)

WidrowHoff.EPOCH = 1

widrow_hoff = WidrowHoff(n_dim=mnist.data.shape[1], n_class=10)

for i in range(100):
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        mnist.data, mnist.target, test_size=0.2, shuffle=True
    )

    # train
    widrow_hoff.train(x_train, y_train)

    # evaluate error rate
    errors = [widrow_hoff.predict(x) != y
              for x, y in zip(list(x_test), list(y_test))]
    error_rate = len(x_test[errors]) / len(x_test)

    print(f"{i} {error_rate}")
