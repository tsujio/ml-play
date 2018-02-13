from matplotlib import pyplot as plt, font_manager
import numpy as np
from sklearn import datasets

dataset = datasets.load_iris()

fp = font_manager.FontProperties(fname=r'C:\Windows\Fonts\meiryo.ttc', size=12)

x_c0 = dataset.data[dataset.target == 0]
x_c1 = dataset.data[dataset.target == 1]
x_c2 = dataset.data[dataset.target == 2]

plt.scatter(x_c0[:, 2], x_c0[:, 3], label='Setosa')
plt.scatter(x_c1[:, 2], x_c1[:, 3], label='Versicolour')
plt.scatter(x_c2[:, 2], x_c2[:, 3], label='Virginica')

plt.xlabel('花弁の長さ', fontproperties=fp)
plt.ylabel('花弁の幅', fontproperties=fp)
plt.title('アヤメ科の花の花弁の大きさに基づいた散布図', fontproperties=fp)

plt.legend()

plt.show()
