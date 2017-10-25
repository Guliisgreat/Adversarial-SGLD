
import torch
from torch.autograd import Variable


class fc(object):
    """docstring for fc"""

    def __init__(self, Hn, input_dim=8 * 8, output_dim=10):
        super(fc, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, Hn),
            torch.nn.ReLU(),
            torch.nn.Linear(Hn, Hn),
            torch.nn.ReLU(),
            torch.nn.Linear(Hn, output_dim),
        )
        self.model = model

    def forward(self, input):
        return self.model(input.view(-1,self.input_dim))

    def parameters(self):
        return self.model.parameters()

    def zero_grad(self):
        self.model.zero_grad()

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def save(self, name):
        dic = {}
        dic['model'] = self.model.state_dict()
        torch.save(dic, name)

    def load(self, name):
        dic = torch.load(name)
        self.model.load_state_dict(dic['model'])

    def type(self, dtype):
        self.model.type(dtype)


class lr(object):
    """docstring for logistic regression model"""

    def __init__(self, input_dim=8 * 8, output_dim=10):
        super(lr, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        models = torch.nn.Sequential(
            torch.nn.Linear(input_dim, output_dim)
        )
        self.model = models

    def forward(self, x):
        y = self.model(x.view(-1, self.input_dim))
        return y

    def parameters(self):
        return self.model.parameters()

    def zero_grad(self):
        self.model.zero_grad()

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def save(self, name):
        dic = {}
        dic['model'] = self.model.state_dict()
        torch.save(dic, name)

    def load(self, name):
        dic = torch.load(name)
        self.model.load_state_dict(dic['model'])

    def type(self, dtype):
        self.model.type(dtype)



# class lr(torch.nn.Module):
#     """docstring for logistic regression model"""
#
#     def __init__(self, input_dim=28 * 28, output_dim=10):
#         super(lr, self).__init__()
#         self.linear = torch.nn.Linear(input_dim, output_dim)
#
#
#     def forward(self, x):
#         y_prediction = self.linear(x).clamp(min=0)
#         return y_prediction


