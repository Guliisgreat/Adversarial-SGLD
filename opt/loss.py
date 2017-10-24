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
        self.CrossEntropyLoss = torch.nn.CrossEntropyLoss()

    def softmax_output(self, inputs):
        prob_inputs = self.softmax(inputs)
        return prob_inputs.data.numpy()

    def log_softmax_output(self, inputs):
        prob_inputs = self.lsoftmax(inputs)
        return prob_inputs.data.numpy()

    def inference_prediction(self, inputs):
        prob_inputs = self.softmax(inputs)
        prediction = prob_inputs.data.numpy().argmax(1)
        return prediction

    def cross_entropy_loss(self, inputs, label):
        prob_inputs = self.softmax(inputs)
        cross_e = self.CrossEntropyLoss(prob_inputs, label)
        return cross_e

    def nll_loss(self, inputs, label):
        prob_inputs = self.lsoftmax(inputs)
        nll = self.nll(prob_inputs, label)
        return nll

        # I think there should be another easy solution to extract gradient data. i will figure it out
    def store_gradient_data(self, inputs, label):
        inputs = inputs.data.clone().type(torch.FloatTensor)
        inputs = Variable(inputs, requires_grad=True)
        prob_inputs = self.lsoftmax(inputs)
        cross_e_loss = self.CrossEntropyLoss(prob_inputs,label)
        cross_e_loss.backward()
        gradient_data = inputs.grad.data
        return gradient_data



