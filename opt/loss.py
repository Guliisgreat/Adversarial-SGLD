import torch
from torch.autograd import Variable
import numpy as np


class CE(object):
    """docstring for CE
    standard classification loss
    """
    def __init__(self):
        super(CE, self).__init__()
        self.lsoftmax = torch.nn.LogSoftmax()
        self.nll = torch.nn.NLLLoss()
        self.CrossEntropyLoss = torch.nn.CrossEntropyLoss()

    def train(self, F, Y_batch):
        tv_F = Variable(F, requires_grad=True)
        tv_Y = Variable(torch.LongTensor(Y_batch.numpy().argmax(1)))
        py_x = self.lsoftmax(tv_F)
        loss = self.nll(py_x, tv_Y)
        ##
        loss.backward()
        G = tv_F.grad.data
        train_pred = py_x.data.numpy().argmax(1)
        return loss.data[0], G, train_pred

    def infer(self, model, X_val, ret_proba=False):
        py_x = self.lsoftmax(model.forward(X_val))
        proba = py_x.data.cpu().numpy()
        val_pred = proba.argmax(1)

        if ret_proba:
            return val_pred, proba
        else:
            return val_pred


    # cross_entropy
    def infer_cross_e(self, model, X_val, Y_val, ret_proba=False):
        py_x = self.lsoftmax(model.forward(X_val))

        tv_Y = Variable(torch.LongTensor(Y_val.argmax(1)))
        loss = self.cross_e(py_x, tv_Y)

        return loss.data[0]

