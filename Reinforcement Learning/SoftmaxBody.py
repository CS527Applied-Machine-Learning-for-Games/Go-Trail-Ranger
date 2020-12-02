import torch.nn as nn
import torch.nn.functional as F

# Define the body of NN
class SoftmaxBody(nn.Module):
    def __init__(self, T):
        super(SoftmaxBody, self).__init__()
        self.T = T

    def forward(self, outputs):
        probabilities = F.softmax(outputs * self.T, dim=0)
        actions = probabilities.multinomial(num_samples=1)
        return actions