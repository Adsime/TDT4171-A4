import skeleton_v2 as s
import numpy as np


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
