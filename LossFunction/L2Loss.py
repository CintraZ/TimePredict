import torch
import torch.nn as nn


class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

        return

    def forward(self, predict, label):
        loss = (predict - label)**2

        return loss.mean()
