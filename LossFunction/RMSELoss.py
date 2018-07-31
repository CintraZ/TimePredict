import torch
import torch.nn as nn


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

        return

    def forward(self, predict, label):
        loss = torch.mean((predict - label)**2)
        loss = torch.sqrt(loss)
        return loss
