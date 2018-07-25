import torch
import utils
import torch.nn as nn

EPS = 10


class LossFunction(nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()

        return

    def forward(self, predict, label):
        predict = utils.unnormalize(predict)
        label = utils.unnormalize(label)
        loss = torch.abs(predict - label) / (label + EPS)

        return loss.mean()
