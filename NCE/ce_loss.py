import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
class CELoss(nn.Module):

    def __init__(self):
        super(CELoss, self).__init__()

    def forward(self, assignments, targets):
        batch_size = assignments.size(0)

        assignments = F.softmax(assignments, dim=1)
        loss = torch.sum(- torch.log(assignments + 1e-6) * targets) / batch_size

        return loss