import os
import torch
import utils
import logging
import TimeDataSet
import LossFunction
import torch.nn as nn
import logging.handlers
import torch.nn.functional as F
from torch.utils.data import DataLoader

EPS = 10

LOG_FILE = 'model.log'

trainDir = 'D:\\WorkFile\\PyTorch\\WorkSpace\\timedata'
testDir = ''
valDir = ''

embeds_dims = [('lineNo', 479, 9), ('busNo', 6366, 13), ('upNo', 2, 1), ('nextSNo', 89, 7),
               ('weekNo', 7, 3), ('timeNo', 24 * 60, 11)]

# Mean distance dictionary
meanDisDict = {}

# Dictionary: line, bus, station
dictList = []

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define Hyper Parameter
num_epochs = 5
num_classes = 1
batch_size = 2
learning_rate = 0.001


# Define Neural Network
class Net(nn.Module):
    def __init__(self, num_classes=1):
        super(Net, self).__init__()

        # Embedding layer
        self.build()

        # Affine operation
        self.fc1 = nn.Linear(45, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, num_classes)

    def build(self):
        for attr, inDim, outDim in embeds_dims:
            self.add_module(attr+'_em', nn.Embedding(inDim, outDim))

    def forward(self, Input):
        # Input affine:line, terminal, up, station, week, time
        Input = torch.squeeze(Input)
        # print('Input size: {}'.format(Input.size()))
        # print('Input size: {}'.format(Input.size()))
        em_lists = []
        for j, (attr, inDim, outDim) in enumerate(embeds_dims):
            embed = getattr(self, attr + '_em')
            intInput = Input[j]
            if j == 0:
                intInput = Input[j].int().item()
                attr_t = dictList[0][intInput]
                attr_t = torch.tensor(attr_t).view(-1, 1)
            elif j == 1:
                intInput = Input[j].int().item()
                attr_t = dictList[1][intInput]
                attr_t = torch.tensor(attr_t).view(-1, 1)
            elif j == 3:
                intInput = Input[j].int().item()
                attr_t = dictList[2][intInput]
                attr_t = torch.tensor(attr_t).view(-1, 1)
            elif j == 2:
                attr_t = Input[j].view(-1, 1)
                attr_t = attr_t[0].float()
            else:
                attr_t = Input[j].view(-1, 1)
            if j != 2:
                attr_t = torch.squeeze(embed(attr_t.long().to(device)))
            em_lists.append(attr_t)

        # Get mean distance and link it to list
        # distTuple = (input[0], input[1], input[2], input[3])
        # dist = meanDisDict[distTuple][2]

        dis = Input[6].view(-1, 1).float()
        em_lists.append(dis[0])
        # print(em_lists)
        output = torch.cat(em_lists, dim=0)
        # print('vector: '.format(output))
        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        return output


model = Net(num_classes).to(device)
print(model)

# loss and optimizer
criterion = LossFunction.LossFunction()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Get mean distance dict
# meanDisDict = utils.GetMeanDisDict()

# Get line, bus, station dictionary
dictList = utils.GetBusLineSDict()
# print(dictList)

# Train the model

model.eval()
for epoch in range(num_epochs):

    for n, (root, dirs, files) in enumerate(os.walk(trainDir)):
        for file in files:
            print('train on file: {}'.format(root + '\\' + file))
            trainDataSet = TimeDataSet.TimeDataSet(file, root, predict=False)
            trainLoader = DataLoader(trainDataSet, batch_size=batch_size, shuffle=True)
            for i, (x, time) in enumerate(trainLoader):
                x = x.to(device)
                # print(x.size())
                time = time.to(device)
                # pass records which distance = 0
                # if x7 == 0:
                #     # print('pass')
                #     pass

                outputs = model(x)
                loss = criterion(outputs, time.float())

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if n % 50 == 49:
            print('util file {}, loss = {:.4f}'.format(file, loss.item()))
    print('Epoch {}/{}, loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
"""
    # Validate model
    for root, dirs, files in os.walk(valDir):
        for file in files:
            print('Validate on file: {}'.format(root + '\\' + file))
            valDataSet = TimeDataSet.TimeDataSet(file, root, predict=False)
            valLoader = DataLoader(trainDataSet, batch_size=batch_size, shuffle=True)


# val

# model.eval()
# with torch.no_grad():

# # Test
# model.eval()
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for time, input in test_loader:
#         predicted = model(input.float())
#         total += time.size(0)

#         correct += ((predicted-time.float())/time.float() <= 0.1).sum().item()

#     print('Test correct rate: {} %'.format(100*correct/total))
"""