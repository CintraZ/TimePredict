import torch
import torch.nn as nn
import Net.Base.Embed
import Net.Base.ResNet
import Net.Base.FullConnect

embeds_dims = [('lineNo', 479, 8), ('busNo', 6366, 16), ('upNo', 2, 2), ('nextSNo', 89, 8),
               ('weekNo', 7, 3), ('timeNo', 24 * 60, 8)]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define Neural Network
class TimePredict(nn.Module):
    def __init__(self, model='fc', num_classes=1):
        super(TimePredict, self).__init__()
        self.model = model
        # Embedding layer
        self.embed = Net.Base.Embed.Net()
        if model == 'fc':
            self.layer = Net.Base.FullConnect.Net(num_classes=num_classes)
        elif model == 'res':
            self.layer = Net.Base.ResNet.ResNet18(num_classes=num_classes)
        # print(self.named_parameters())
        self.init_weight()

    def init_weight(self):
        for name, param in self.named_parameters():
            if name.find('.bias') != -1:
                param.data.fill_(0)
            elif name.find('.weight') != -1:
                nn.init.xavier_uniform_(param.data)

    def forward(self, Input):
        # Input affine:line, terminal, up, station, week, time
        output = self.embed(Input)
        output = self.layer(output)

        return output


# model = Net(1)
# # print(model)
# params = model.state_dict()
# for k, v in params.items():
#     print(k)
#
# print(params['net.fc1.weight'])
# print(params['net.fc1.bias'])
