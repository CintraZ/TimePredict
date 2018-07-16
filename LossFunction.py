import torch
import torch.nn as nn


class LossFunction(nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()

        return

    def forward(self, predict, label):
        loss = torch.abs(predict - label) / (label + 10)

        return loss.mean()
