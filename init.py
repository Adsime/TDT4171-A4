import skeleton_v2 as s
import functions as f
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

def task_1_1():
    w1 = np.arange(-6, 6, 1)
    w2 = np.arange(-6, 6, 1)

    xx, yy = np.meshgrid(w1, w2)
    zz = []
    for x, y in zip(xx, yy):
        a = []
        for i, j in zip(x, y):
            a.append(f.simple_loss([i, j]))
        zz.append(a)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel("w1")
    ax.set_ylabel("w2")
    ax.scatter(xx, yy, zz)
    plt.show()


def task_1_3(alpha, epochs):
    w1, w2 = [[0], [0]]
    for x in range(0, epochs):
        for i, j in zip(w1, w2):
            nw1, nw2 = f.derivative(w1, w2)
            w1 = f.update(i, nw1[0], alpha)
            w2 = f.update(j, nw2[0], alpha)
    return f.simple_loss([w1[0], w2[0]])


def task_1_3_plot():
    alphas = [0.0001, 0.01,  0.1, 1, 10, 100]
    err = []
    for alpha in alphas:
        err.append(task_1_3(alpha, 100))
    plt.plot(alphas, err)
    plt.show()

task_1_3_plot()
task_1_3(0.1, 100)
task_1_1()

