import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

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
netz = netz.cuda()

print(netz)

input = Variable(torch.randn(3,3))
input = input.cuda()
print(input)

output = netz(input)
print(output)

X = [0,0,1]
target = Variable(torch.Tensor([X for _ in range(3)]))
target = target.cuda()
criterion = nn.MSELoss()

loss = criterion(output,target)

print(loss)

netz.zero_grad()
loss.backward()
optimizer = optim.SGD(netz.parameters(), lr= 0.01)
optimizer.step()
