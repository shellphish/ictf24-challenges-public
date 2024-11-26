import torch

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(133, 133)
        self.linear2 = torch.nn.Linear(133, 133)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x