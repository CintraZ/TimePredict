import torch
import torch.nn as nn
from collections import OrderedDict

embeds_dims = [('lineNo', 479, 8), ('busNo', 6366, 16), ('upNo', 2, 2), ('nextSNo', 89, 8),
               ('weekNo', 7, 3), ('timeNo', 24 * 60, 8)]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

count = 0


# Define Neural Network
class Net(nn.Module):
    def __init__(self, num_classes=1):
        super(Net, self).__init__()

        # Embedding layer
        self.build()

        # Affine operation
        self.net = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(46, 1024)),
            # ('bn1', nn.BatchNorm1d(1024)),
            ('relu1', nn.LeakyReLU()),
            ('fc2', nn.Linear(1024, 1024)),
            # ('bn2', nn.BatchNorm1d(1024)),
            ('relu2', nn.LeakyReLU()),
            ('fc3', nn.Linear(1024, 512)),
            # ('bn3', nn.BatchNorm1d(512)),
            ('relu3', nn.LeakyReLU()),
            ('fc4', nn.Linear(512, 128)),
            # ('bn4', nn.BatchNorm1d(128)),
            ('relu4', nn.LeakyReLU()),
            ('fc5', nn.Linear(128, num_classes))

        ]))
        # print(self.named_parameters())
        self.init_weight()

    def init_weight(self):
        for name, param in self.named_parameters():
            if name.find('.bias') != -1:
                param.data.fill_(0)
            elif name.find('.weight') != -1:
                nn.init.xavier_uniform_(param.data)

    def build(self):
        for attr, inDim, outDim in embeds_dims:
            self.add_module(attr + '_em', nn.Embedding(inDim, outDim))

    def forward(self, Input):
        global count
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

        # Visualization????
        # if count == 0:
        #     count += 1
        #     tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        #     X_tsne = tsne.fit_transform(output.cpu().detach().numpy())
        #     fig2 = plt.figure(2)
        #     axes = fig2.subplots(1, 1)
        #     x_list = []
        #     y_list = []
        #     for i in range(X_tsne.shape[0]):
        #         x_list.append(X_tsne[i, 0])
        #         y_list.append(X_tsne[i, 1])
        #         # axes.scatter(X_tsne[i, 0], X_tsne[i, 1], c='b')
        #     colors = np.arctan2(y_list, x_list)
        #     axes.scatter(x_list, y_list, c=colors)
        #     fig2.savefig('emm.png')
        output = self.net(output)

        return output


# model = Net(1)
# # print(model)
# params = model.state_dict()
# # for k, v in params.items():
#     # print(k)
#
# print(params['net.fc1.weight'])
# print(params['net.fc1.bias'])
