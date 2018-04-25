import skeleton_v2 as s
import functions as f
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import random
import pandas as pd

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
        err.append(task_1_3(alpha, 10))
    plt.plot(alphas, err)
    plt.show()

def task_2_1():
    x, y = f.from_file(f.db_sep_train)
    xt, yt = f.from_file(f.db_sep_test)

    s.train_and_plot(x, y, xt, yt, s.batch_train_w, 0.6, 10)


def task_2_3():
    sep_train = [f.from_file(f.db_sep_train), f.from_file(f.ds_sep_train)]
    sep_test = [f.from_file(f.db_sep_test), f.from_file(f.ds_sep_test)]
    nonsep_train = [f.from_file(f.db_nonsep_train), f.from_file(f.ds_nonsep_train)]
    nonsep_test = [f.from_file(f.db_nonsep_test), f.from_file(f.ds_nonsep_test)]
    f.perceptron_test(sep_train, sep_test, s.stochast_train_w, "separable")
    f.perceptron_test(nonsep_train, nonsep_test, s.stochast_train_w, "non separable")
    #f.perceptron_test(sep_train, sep_test, s.batch_train_w, "separable")
    #f.perceptron_test(nonsep_train, nonsep_test, s.batch_train_w, "non separable")

    s.train_and_plot(sep_train[0][0], sep_train[0][1], sep_train[1][0], sep_train[1][1], s.stochast_train_w, 0.3, 100)

    coords = []
    time = []
    name = ""
    for i in np.arange(10, 500, 1):
        res = s.train_and_test(sep_train[0][0], sep_train[0][1], sep_train[1][0], sep_train[1][1], s.stochast_train_w, 0.3, i, False)
        coords.append([res[2], res[1]])
        time.append(res[3])
    # error plot
    plt.plot([x for x, y in coords], [y for x, y in coords])
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.title("Error for x iterations")
    plt.figure()
    # time plot
    plt.plot([x for x, y in coords], time)
    plt.xlabel("Iterations")
    plt.ylabel("Seconds")
    plt.title("Seconds for x iterations")

    coords = np.array(coords)
    time = np.array(time)
    data_test = pd.DataFrame(np.hstack((coords, time.reshape(coords.shape[0], 1))), columns=['iterations', 'error', 'seconds'])
    data_test.plot(kind='scatter', x='iterations', y='error', c='seconds', cmap='copper', edgecolors='black', title=name)
    plt.show()

task_1_1()
task_1_3_plot()
task_2_1()
task_2_3()