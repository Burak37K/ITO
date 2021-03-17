import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

classificationModel = nn.Sequential(
    nn.Linear(28 * 28, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

classificationModelParameters = classificationModel.parameters()

classificationOptimizer = optim.SGD(classificationModelParameters, lr=1e-2)

classificationLoss = nn.CrossEntropyLoss()

losses = list()

torch.randn(5).cuda()

train_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
train,val = random_split(train_data, [55000,5000])
train_loader = DataLoader(train, batch_size=32)
values_loader = DataLoader(val, batch_size=32)


nb_epochs = 5
for epoch in range(nb_epochs):
    for batch in train_loader:
        AnyVariableX, AnyVariableY = batch

        SizeFromX = AnyVariableX.size(0)
        ViewOfX = AnyVariableX.view(SizeFromX, -1).cuda()

        logitValue = classificationModel(ViewOfX)

        ComputeTheObjectFunction = classificationLoss(logitValue, AnyVariableY.cuda())

        classificationModel.zero_grad()  #cleans the gradient

        ComputeTheObjectFunction.backward()     #backward accumulate the partial derivatives of ComputeThe.. params

        classificationOptimizer.step()

        losses.append(logitValue.item())

        print(f'Eporch{epoch +1}, train loss: { torch.tensor(losses).mean(): .2f}' )