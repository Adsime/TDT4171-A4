import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
import time as t

def logistic_z(z): 
    return 1.0/(1.0+np.exp(-z))


def logistic_wx(w,x): 
    return logistic_z(np.inner(w,x))


def classify(w,x):
    x=np.hstack(([1],x))
    return 0 if (logistic_wx(w,x)<0.5) else 1


def loss(i, x, w, y):
    return (logistic_wx(w, x) - y) * x[i] * np.exp(-np.inner(w, x)) * np.square(logistic_wx(w, x))


#x_train = [number_of_samples,number_of_features] = number_of_samples x \in R^number_of_features
def stochast_train_w(x_train,y_train,learn_rate=0.1,niter=1000):
    x_train=np.hstack((np.array([1]*x_train.shape[0]).reshape(x_train.shape[0],1),x_train))
    dim=x_train.shape[1]
    num_n=x_train.shape[0]
    w = np.random.rand(dim)
    index_lst = []
    for it in range(niter):
        if len(index_lst) == 0:
            index_lst = random.sample(range(num_n), k=num_n)
        xy_index = index_lst.pop()
        x = x_train[xy_index, :]
        y = y_train[xy_index]
        for i in range(dim):
            update_grad = loss(i, x, w, y) ### something needs to be done here
            w[i] = w[i] - (learn_rate * update_grad) ### something needs to be done here
    return w


def batch_train_w(x_train,y_train,learn_rate=0.1,niter=1000):
    x_train = np.hstack((np.array([1]*x_train.shape[0]).reshape(x_train.shape[0],1),x_train))
    dim = x_train.shape[1]
    num_n = x_train.shape[0]
    w = np.random.rand(dim)
    for it in range(niter):
        for i in range(dim):
            update_grad = 0.0
            for n in range(num_n):
                update_grad += loss(i, x_train[n], w, y_train[n])# something needs to be done here
            w[i] = w[i] - (learn_rate * (update_grad/num_n))
    return w


def train_and_plot(xtrain,ytrain,xtest,ytest,training_method,learn_rate=0.1,niter=10):
    #train data
    data = pd.DataFrame(np.hstack((xtrain,ytrain.reshape(xtrain.shape[0],1))),columns=['x','y','lab'])
    data.plot(kind='scatter',x='x',y='y',c='lab',cmap='copper',edgecolors='black', title='train data')

    #train weights
    w = training_method(xtrain, ytrain, learn_rate, niter)
    error = []
    y_est = []
    for i in range(len(ytest)):
        error.append(np.abs(classify(w,xtest[i])-ytest[i]))
        y_est.append(classify(w,xtest[i]))
    y_est=np.array(y_est)
    data_test = pd.DataFrame(np.hstack((xtest,y_est.reshape(xtest.shape[0],1))),columns=['x','y','lab'])
    data_test.plot(kind='scatter', x='x', y='y', c='lab', cmap='copper', edgecolors='black', title='test data')
    print("error=", np.mean(error))
    plt.show()
    return w


def train_and_test(xtrain,ytrain,xtest,ytest,training_method,learn_rate=0.1,niter=10, print_res=True):
    tt = t.time()
    w = training_method(xtrain, ytrain, learn_rate, niter)
    tt = (t.time() - tt)
    error = []
    y_est = []
    for i in range(len(ytest)):
        error.append(np.abs(classify(w, xtest[i]) - ytest[i]))
        y_est.append(classify(w, xtest[i]))
    if print_res:
        print("Epochs: " + niter.__str__())
        print("Leaning rate: " + learn_rate.__str__())
        print("Training time " + tt.__str__())
        print("Error: " + np.mean(error).__str__() + "\n\n")
    return [training_method.__name__, np.mean(error), float(niter), tt]