import numpy as np

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def int_to_onehot(y, num_labels):
    arr = np.zeros((y.shape[0], num_labels))
    for i, val in enumerate(y):
        arr[i, val] = 1

    return arr
