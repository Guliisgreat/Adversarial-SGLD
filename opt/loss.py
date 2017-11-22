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
        self.softmax = torch.nn.Softmax()
        self.nll = torch.nn.NLLLoss()
        self.CEL = torch.nn.CrossEntropyLoss()

    def CrossEntropyLoss(self, outputs, labels):
        return self.CEL(outputs.type(torch.FloatTensor), labels)
    def softmax_output(self, inputs):
        prob_inputs = self.softmax(inputs)
        return prob_inputs

    def log_softmax_output(self, inputs):
        prob_inputs = self.lsoftmax(inputs)
        return prob_inputs

    def inference_prediction(self, inputs):
        prob_inputs = self.softmax(inputs)
        prediction = prob_inputs.data.cpu().numpy().argmax(1)
        return prediction

    def cross_entropy_loss(self, inputs, label):
        prob_inputs = self.softmax(inputs)
        cross_e = self.CEL(prob_inputs, label)
        return cross_e

    def nll_loss(self, inputs, label):
        prob_inputs = self.lsoftmax(inputs.type(torch.FloatTensor))
        nll = self.nll(prob_inputs, label)
        return nll

        # I think there should be another easy solution to extract gradient data. i will figure it out
    def store_gradient_data(self, inputs, label):
        inputs = inputs.data.clone().type(torch.FloatTensor)
        inputs = Variable(inputs, requires_grad=True)
        prob_inputs = self.lsoftmax(inputs)
        nll_loss = self.nll(prob_inputs,label)
        nll_loss.backward()
        gradient_data = inputs.grad.data
        return gradient_data

    def train(self, F, Y_batch):
        tv_F = Variable(F, requires_grad=True)
        tv_Y = Variable(torch.LongTensor(Y_batch.numpy().argmax(1)))
        py_x = self.softmax(tv_F)
        loss = self.CEL(py_x, tv_Y)
        ##
        loss.backward()
        G = tv_F.grad.data
        train_pred = py_x.data.numpy().argmax(1)
        return loss.data[0], G, train_pred

    def infer(self, model, X_val, Y_val, ret_proba=False):
        py_x = self.softmax(model.forward(X_val))
        proba = py_x.data.cpu().numpy()
        val_pred = proba.argmax(1)

        tv_Y = Variable(torch.LongTensor(Y_val.data.numpy().argmax(1)))

        ce_loss = self.CEL( py_x,tv_Y)
        if ret_proba:
            return val_pred, ce_loss, proba
        else:
            return val_pred, ce_loss
