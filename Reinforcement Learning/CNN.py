import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# AI Brain - Using CNN
class CNN(nn.Module):
    def __init__(self, number_actions):
        super(CNN, self).__init__()
        self.convolution1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.convolution2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.convolution3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.out_neurons = self.count_neurons((1, 128, 128))
        self.lstm = nn.LSTMCell(self.out_neurons, 256)
        # self.fc1 = nn.Linear(in_features = self.count_neurons((1, 256, 256)), out_features = 62)
        self.fc2 = nn.Linear(in_features=256, out_features=number_actions)
        self.apply(initialize_weights)
        self.fc2.bias.data.fill_(0)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        self.train()

    def count_neurons(self, image_dim):
        x = Variable(torch.rand(1, *image_dim))
        x = F.elu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.elu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.elu(F.max_pool2d(self.convolution3(x), 3, 2))
        return x.data.view(1, -1).size(1)

    def forward(self, x, hidden=None):
        x = x.cpu()
        if isinstance(hidden, tuple):
            hidden = (hidden[0].cpu(), hidden[1].cpu())
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        x = x.view(-1, self.out_neurons)
        hx, cx = self.lstm(x, hidden)
        x = hx
        # x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x, (hx, cx)


# Initializing the weights of NN in an optimal way for learning
def initialize_weights(model):
    class_name = model.__class__.__name__

    if class_name.find('Conv') != -1:
        weight_shape = list(model.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        model.weight.data.uniform_(-w_bound, w_bound)
        model.bias.data.fill_(0)
    elif class_name.find('Linear') != -1:
        weight_shape = list(model.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        model.weight.data.uniform_(-w_bound, w_bound)
        model.bias.data.fill_(0)