import torch
from torch.autograd import Variable
import torch.nn
import numpy as np


def loss(y_pred, y):
    return (y_pred.item() - y.item())**2


''' Returns the mean accuracy on the test set, given a model '''


def eval_on_test(test_data_X, test_data_Y, model_fn):
    mse = 0
    Y = test_data_Y
    X = test_data_X
    Y = torch.tensor(Y.values)
    X = torch.tensor(np.asarray(X))
    Y = torch.tensor(np.asarray(Y))
    for x, y in zip(X, Y):
        x, y = Variable(x), Variable(y)

        if torch.cuda.is_available():
            x, y = x.cuda(), y.cuda()
        mse += loss(model_fn(x.float()), y)
    return round(mse / float(len(test_data_X)), 3)


''' Converts a list of (x, x) pairs into two Tensors '''


def into_tensor(data, into_vars=True):
    X1 = [x[0] for x in data]
    X2 = [x[1] for x in data]
    if torch.cuda.is_available():
        return Variable(torch.stack(X1)).cuda(), Variable(torch.stack(X2)).cuda()
    return Variable(torch.stack(X1)), Variable(torch.stack(X2))
