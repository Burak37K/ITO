import torch
import torch.nn as nn
import torch.nn.functional as F

class myNet(nn.Module):
    def __init__(self):
        super(myNet, self).__init__()
        self.lin1 = nn.Linear(3,3)
        self.lin2 = nn.Linear(3,3)


    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

    def num_flat_features(self,x):
        size = x.size()[1:]
        num = 1
        for i in size:
            num *= 1


netz = myNet()

print(netz)

