import skeleton_v2 as s
import numpy as np
from matplotlib import pyplot as plt

path = "data/"
db_nonsep_test = path + "data_big_nonsep_test.csv"
db_nonsep_train = path + "data_big_nonsep_train.csv"
db_sep_test = path + "data_big_separable_test.csv"
db_sep_train = path + "data_big_separable_train.csv"
ds_nonsep_test = path + "data_small_nonsep_test.csv"
ds_nonsep_train = path + "data_small_nonsep_train.csv"
ds_sep_test = path + "data_small_separable_test.csv"
ds_sep_train = path + "data_small_separable_train.csv"


def from_file(file):
    data = []
    target = []
    for line in open(file).readlines():
        l = [float(x) for x in line.replace("\n", "").split("\t")]
        target.append(l.pop(-1))
        data.append(l)
    return [np.array(data), np.array(target)]


def simple_loss(w):
    return np.square(s.logistic_wx(w, [1, 0]) - 1) + \
           np.square(s.logistic_wx(w, [0, 1])) + \
           np.square(s.logistic_wx(w, [1, 1]) - 1)


def derivative(w1_arr, w2_arr):
    x = [[1, 0], [0, 1], [1, 1]]
    new_w1 = []
    new_w2 = []
    for w1, w2 in zip(w1_arr, w2_arr):
        new_w1.append(partial_deriv(x[0][0], x[0], [w1, w2], True) +
                      partial_deriv(x[1][0], x[1], [w1, w2], False) +
                      partial_deriv(x[2][0], x[2], [w1, w2], True))
        new_w2.append(partial_deriv(x[0][1], x[0], [w1, w2], True) +
                      partial_deriv(x[1][1], x[1], [w1, w2], False) +
                      partial_deriv(x[2][1], x[2], [w1, w2], True))
    return [new_w1, new_w2]


def partial_deriv(x_i, x, w, include):
    v = np.dot(np.transpose(w), x)
    nomin = (2 * x_i * np.exp((2 if include else 1) * (-v)))
    res = (nomin/(1 + np.exp(-v))**3)
    return -res if include else res


def update(old_w, new_w, alpha):
    return [old_w - (alpha * new_w)]


def perceptron_test(train_data, test_data, method, separable):
    print("---- Start test: " + separable + " data - Method: " + method.__name__ + " ----")
    data = []
    for train, test in zip(train_data, test_data):
        data.append(s.train_and_test(train[0], train[1], test[0], test[1], method, 0.3, 100))
    print("---- End test: " + separable + " data - Method: " + method.__name__ + " ----\n\n")
    return data


