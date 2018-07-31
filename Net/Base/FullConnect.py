import torch
import torch.nn as nn
import torch.nn.functional as F

embeds_dims = [('lineNo', 479, 8), ('busNo', 6366, 16), ('upNo', 2, 2), ('nextSNo', 89, 8),
               ('weekNo', 7, 3), ('timeNo', 24 * 60, 8)]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define Neural Network
class Net(nn.Module):
    def __init__(self, num_classes=1):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(46, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 128)
        self.fc5 = nn.Linear(128, num_classes)

    def forward(self, Input):
        output = F.leaky_relu(self.fc1(Input))
        # output = F.leaky_relu(self.fc2(output))
        output = F.leaky_relu(self.fc3(output))
        output = F.leaky_relu(self.fc4(output))

        output = self.fc5(output)
        # output = self.net(output)

        return output


# model = Net(1)
# # print(model)
# params = model.state_dict()
# for k, v in params.items():
#     print(k)
#
# print(params['net.fc1.weight'])
# print(params['net.fc1.bias'])
