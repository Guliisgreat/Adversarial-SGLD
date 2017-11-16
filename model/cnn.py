
import torch 
import torchvision.models as tvm

class ResNet(object):
    def __init__(self, suffix=18):
        super(ResNet, self).__init__()
        self.model = eval('tvm.resnet%g'%suffix)()
    def eval(self):
        self.model.eval()
        
    def train(self):
        self.model.train()
    def parameters(self):
        return list(self.model.parameters())
    def zero_grad(self):
        self.model.zero_grad()
    def forward(self, input):
        input = input.permute(0,3,1,2)
        tmp = self.model(input)
        return tmp

    def save(self, name):
        dic = {}
        dic['model'] = self.model.state_dict()
        torch.save(dic, name)

    def load(self, name):
        dic = torch.load(name)
        self.model.load_state_dict(dic['model'])
    def type(self, dtype):
        self.model.type(dtype)

class cnn_base(object):
    def __init__(self):
        super(cnn_base, self).__init__()
    def eval(self):
        self.fc.eval()
        self.conv.eval()
    def train(self):
        self.fc.train()
        self.conv.train()
    def parameters(self):
        return list(self.conv.parameters())+list(self.fc.parameters())
    def zero_grad(self):
        self.fc.zero_grad()
        self.conv.zero_grad()
    def forward(self, input):
        input = input.permute(0,3,1,2)
        tmp = self.conv(input)
        tmp = tmp.view(-1,self.fc_dim)
        return self.fc(tmp)

    def save(self, name):
        dic = {}
        dic['conv'] = self.conv.state_dict()
        dic['fc'] = self.fc.state_dict()
        torch.save(dic, name)

    def load(self, name):
        dic = torch.load(name)
        self.conv.load_state_dict(dic['conv'])
        self.fc.load_state_dict(dic['fc'])
    def type(self, dtype):
        self.conv.type(dtype)
        self.fc.type(dtype)

    def state_dict(self):
        return self.conv.state_dict(), self.fc.state_dict()

    def load_state_dict(self, sd):
        self.conv.load_state_dict(sd[0])
        self.fc.load_state_dict(sd[1])
class cnn(cnn_base):
    """docstring for fc"""
    def __init__(self, Hn, input_dim=[28,28,1],output_dim=10):
        super(cnn, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(input_dim[-1], Hn, 3,padding=1),
            torch.nn.MaxPool2d((2,2)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(Hn, Hn, 3,padding=1),
            torch.nn.MaxPool2d((2,2)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(Hn, Hn, 3,padding=1),
            torch.nn.MaxPool2d((2,2)),
            torch.nn.ReLU(),
        )
        D = input_dim[0]
        for _ in xrange(3):
            D = D//2

        self.fc_dim = Hn * D*D
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.fc_dim, Hn),
            torch.nn.ReLU(),
            torch.nn.Linear(Hn, output_dim),
        )

class dcgan_disc(cnn_base):
    """docstring for fc"""
    def __init__(self, Hn, input_dim=[28,28,1],output_dim=10):
        super(dcgan_disc, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(input_dim[-1], Hn, 4,2,padding=1),
            torch.nn.LeakyReLU(.2),
            torch.nn.BatchNorm2d(Hn),
            torch.nn.Conv2d(Hn, Hn*2, 4,2,padding=1),
            torch.nn.LeakyReLU(.2),
            torch.nn.BatchNorm2d(Hn*2),
            torch.nn.Conv2d(Hn*2, Hn*4, 4,2,padding=1),
            torch.nn.LeakyReLU(.2),
            torch.nn.BatchNorm2d(Hn*4),
            torch.nn.Conv2d(Hn*4, 2, 4,1,padding=0),
            torch.nn.LeakyReLU(.2),
        )
        # D = input_dim[0]
        # for _ in xrange(4):
        #     D = D//2

        self.fc_dim = 2
        self.fc = torch.nn.Sequential(
            # torch.nn.Linear(self.fc_dim, Hn),
            # torch.nn.LeakyReLU(.2),
            torch.nn.Linear(self.fc_dim, output_dim),
        )
    
    
    


class cnn2(cnn_base):
    """docstring for fc"""
    def __init__(self, Hn=1, input_dim=[28,28,1],output_dim=10,dropout=0):
        super(cnn2, self).__init__()
        raise() ### even without dropout it didn't work...
                ### it used to work with 2 conv layers each, 
                ### ....?
        hid1=48
        hid2=96
        hid3=96
        hidfc=256
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(input_dim[-1], Hn*hid1, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(Hn*hid1, Hn*hid1, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(Hn*hid1, Hn*hid1, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2,2)),
            torch.nn.Dropout(dropout),
            torch.nn.Conv2d(Hn*hid1, Hn*hid2, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(Hn*hid2, Hn*hid2, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(Hn*hid2, Hn*hid2, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2,2)),
            torch.nn.Dropout(dropout),
            torch.nn.Conv2d(Hn*hid2, Hn*hid3, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(Hn*hid3, Hn*hid3, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(Hn*hid3, Hn*hid3, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2,2)),
            torch.nn.Dropout(dropout),
            #
        )
        D = input_dim[0]
        for _ in xrange(3):
            D = D//2

        self.fc_dim = Hn*hid3 * D*D
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.fc_dim, hidfc),
            torch.nn.ReLU(),
            torch.nn.Dropout(.1),
            torch.nn.Linear(hidfc, output_dim),
            # torch.nn.Tanh(),
        )


class cnn_bn(cnn_base):
    """docstring for fc"""
    def __init__(self, Hn=1, input_dim=[28,28,1],output_dim=10):
        super(cnn_bn, self).__init__()
        hid1=48
        hid2=96
        hid3=144
        hidfc=256
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(input_dim[-1], Hn*hid1, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(Hn*hid1, Hn*hid1, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(Hn*hid1, Hn*hid1, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2,2)),
            torch.nn.BatchNorm2d(Hn*hid1),
            torch.nn.Conv2d(Hn*hid1, Hn*hid2, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(Hn*hid2, Hn*hid2, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(Hn*hid2, Hn*hid2, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2,2)),
            torch.nn.BatchNorm2d(Hn*hid2),
            torch.nn.Conv2d(Hn*hid2, Hn*hid3, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(Hn*hid3, Hn*hid3, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(Hn*hid3, Hn*hid3, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2,2)),
            torch.nn.BatchNorm2d(Hn*hid3),
            #
        )
        D = input_dim[0]
        for _ in xrange(3):
            D = D//2

        self.fc_dim = Hn*hid3 * D*D
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.fc_dim, hidfc),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidfc),
            torch.nn.Linear(hidfc, output_dim),
            # torch.nn.Tanh(),
        )


class cnn_bn_db(cnn_base):
    """docstring for fc"""
    def __init__(self, Hn=1, input_dim=[28,28,1],output_dim=10):
        super(cnn_bn_db, self).__init__()
        hid1=32
        hid2=32
        hid3=64
        hidfc=64
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(input_dim[-1], Hn*hid1, 3,padding=1),
            torch.nn.ReLU(),
            # torch.nn.Conv2d(Hn*hid1, Hn*hid1, 3,padding=1),
            # torch.nn.ReLU(),
            # torch.nn.Conv2d(Hn*hid1, Hn*hid1, 3,padding=1),
            # torch.nn.ReLU(),
            torch.nn.MaxPool2d((2,2)),
            torch.nn.BatchNorm2d(Hn*hid1),
            torch.nn.Conv2d(Hn*hid1, Hn*hid2, 3,padding=1),
            # torch.nn.ReLU(),
            # torch.nn.Conv2d(Hn*hid2, Hn*hid2, 3,padding=1),
            # torch.nn.ReLU(),
            # torch.nn.Conv2d(Hn*hid2, Hn*hid2, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2,2)),
            torch.nn.BatchNorm2d(Hn*hid2),
            torch.nn.Conv2d(Hn*hid2, Hn*hid3, 3,padding=1),
            # torch.nn.ReLU(),
            # torch.nn.Conv2d(Hn*hid3, Hn*hid3, 3,padding=1),
            # torch.nn.ReLU(),
            # torch.nn.Conv2d(Hn*hid3, Hn*hid3, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2,2)),
            torch.nn.BatchNorm2d(Hn*hid3),
            #
        )
        D = input_dim[0]
        for _ in xrange(3):
            D = D//2

        self.fc_dim = Hn*hid3 * D*D
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.fc_dim, hidfc),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidfc),
            torch.nn.Linear(hidfc, output_dim),
            # torch.nn.Tanh(),
        )


class cnn_globe(cnn_base):
    """popular architecture... specifically following ladder net
    """
    def __init__(self, Hn=1, input_dim=[28,28,1],output_dim=10, dropout=0.2, softmax=False):
        super(cnn_globe, self).__init__()
        hid1=int(96*Hn)
        hid2=int(192*Hn)
        hidfc=128
        self.softmax = softmax
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv = torch.nn.Sequential(
            torch.nn.Dropout(.2),
            torch.nn.Conv2d(input_dim[-1], hid1, 3,padding=1),
            torch.nn.BatchNorm2d(hid1),
            torch.nn.LeakyReLU(.2),
            torch.nn.Conv2d(hid1, hid1, 3,padding=1),
            torch.nn.BatchNorm2d(hid1),
            torch.nn.LeakyReLU(.2),
            torch.nn.Conv2d(hid1, hid1,3,padding=1),
            torch.nn.BatchNorm2d(hid1),
            torch.nn.LeakyReLU(.2),
            torch.nn.MaxPool2d((2,2),stride=2),
            torch.nn.BatchNorm2d(hid1),
            #
            torch.nn.Dropout(dropout),
            torch.nn.Conv2d(hid1, hid2, 3,padding=1),
            torch.nn.BatchNorm2d(hid2),
            torch.nn.LeakyReLU(.2),
            torch.nn.Conv2d(hid2, hid2, 3,padding=1),
            torch.nn.BatchNorm2d(hid2),
            torch.nn.LeakyReLU(.2),
            torch.nn.Conv2d(hid2, hid2, 3,padding=1),
            torch.nn.BatchNorm2d(hid2),
            torch.nn.LeakyReLU(.2),
            torch.nn.MaxPool2d((2,2),stride=2),
            torch.nn.BatchNorm2d(hid2),
            #
            torch.nn.Dropout(.3), # was 0.3 before when hitting baseline
            torch.nn.Conv2d(hid2, hid2, 3,padding=0),
            torch.nn.BatchNorm2d(hid2),
            torch.nn.LeakyReLU(.2),
            ## 1x1 convs on 6x6 images
            torch.nn.Conv2d(hid2, hid2, 1,padding=0),
            torch.nn.BatchNorm2d(hid2),
            torch.nn.LeakyReLU(.2),
            torch.nn.Conv2d(hid2, self.output_dim, 1,padding=0),
            torch.nn.BatchNorm2d(self.output_dim),
            torch.nn.LeakyReLU(.2),
        )
    def forward(self, input):
        N = input.size()[0]
        input = input.permute(0,3,1,2)
        tmp = self.conv(input)
        ## (N, output_dim, 6,6) -> (N, , -1)
        tmp = tmp.view(N, self.output_dim, -1)
        ## average pool across channels
        tmp = tmp.mean(-1).view(N, self.output_dim)
        return tmp
    def eval(self):
        self.conv.eval()
    def train(self):
        self.conv.train()
    def parameters(self):
        return list(self.conv.parameters())
    def zero_grad(self):
        self.conv.zero_grad()
    def save(self, name):
        dic = {}
        dic['conv'] = self.conv.state_dict()
        torch.save(dic, name)

    def load(self, name):
        dic = torch.load(name)
        self.conv.load_state_dict(dic['conv'])
    def type(self, dtype):
        self.conv.type(dtype)

    def state_dict(self):
        return self.conv.state_dict()

    def load_state_dict(self, sd):
        self.conv.load_state_dict(sd)

class cnn_dngo(cnn_base):
    """architecture of the discriminator of improved gan paper
        only difference is that I'm not using leakyrelu
    """
    def __init__(self, Hn=1, input_dim=[28,28,1],output_dim=10, dropout=0.2):
        super(cnn_dngo, self).__init__()
        hid1=96
        hid2=192
        hidfc=128
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv = torch.nn.Sequential(
            torch.nn.Dropout(.2),
            torch.nn.Conv2d(input_dim[-1], Hn*hid1, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(Hn*hid1),
            torch.nn.Conv2d(Hn*hid1, Hn*hid1, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(Hn*hid1),
            torch.nn.MaxPool2d((3,3),stride=2),
            torch.nn.Dropout(dropout),
            torch.nn.Conv2d(Hn*hid1, Hn*hid2, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(Hn*hid2),
            torch.nn.Conv2d(Hn*hid2, Hn*hid2, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(Hn*hid2),
            torch.nn.Conv2d(Hn*hid2, Hn*hid2, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(Hn*hid2),
            torch.nn.MaxPool2d((3,3),stride=2),
            torch.nn.Dropout(dropout),
            torch.nn.Conv2d(Hn*hid2, Hn*hid2, 3,padding=0),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(Hn*hid2),
            ## 1x1 convs on 6x6 images
            torch.nn.Conv2d(Hn*hid2, Hn*hid2, 1,padding=0),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(Hn*hid2),
            torch.nn.Conv2d(Hn*hid2, output_dim, 1,padding=0),
        )
        self.output_dim = output_dim
    def forward(self, input):
        N = input.size()[0]
        input = input.permute(0,3,1,2)
        tmp = self.conv(input)
        ## (N, output_dim, 6,6) -> (N, , -1)
        tmp = tmp.view(N, self.output_dim, -1)
        ## average pool across channels
        tmp = tmp.mean(-1).view(N, self.output_dim)
        return tmp
    def eval(self):
        self.conv.eval()
    def train(self):
        self.conv.train()
    def parameters(self):
        return list(self.conv.parameters())
    def zero_grad(self):
        self.conv.zero_grad()
    def save(self, name):
        dic = {}
        dic['conv'] = self.conv.state_dict()
        torch.save(dic, name)

    def load(self, name):
        dic = torch.load(name)
        self.conv.load_state_dict(dic['conv'])