import torch
import torch.nn as nn

embeds_dims = [('lineNo', 479, 8), ('busNo', 6366, 16), ('upNo', 2, 2), ('nextSNo', 89, 8),
               ('weekNo', 7, 3), ('timeNo', 24 * 60, 8)]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define Neural Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Embedding layer
        self.build()

    def build(self):
        for attr, inDim, outDim in embeds_dims:
            self.add_module(attr + '_em', nn.Embedding(inDim, outDim))

    def forward(self, Input):
        # Input affine:line, terminal, up, station, week, time
        em_lists = []
        for j, (attr, inDim, outDim) in enumerate(embeds_dims):
            embed = getattr(self, attr + '_em')
            attr_t = Input[j].view(-1, 1)
            attr_t = embed(attr_t.long().to(device))

            attr_t = attr_t.view(attr_t.size(0), -1)
            em_lists.append(attr_t)

        # Get mean distance and link it to list
        # distTuple = (input[0], input[1], input[2], input[3])
        # dist = meanDisDict[distTuple][2]

        dis = Input[6].view(Input[6].size(0), -1).float()
        em_lists.append(dis)

        output = torch.cat(em_lists, dim=1)

        return output


# model = Net(1)
# # print(model)
# params = model.state_dict()
# for k, v in params.items():
#     print(k)
#
# print(params['net.fc1.weight'])
# print(params['net.fc1.bias'])
