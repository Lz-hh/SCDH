from torch import nn
import torch
class NCESoftmaxLoss(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""
    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        # self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = x.softmax(1)
        return -x[:, 0].log().mean()



class ModifiedNCESoftmaxLoss(nn.Module):
    def __init__(self):
        super(ModifiedNCESoftmaxLoss, self).__init__()

    def forward(self, pos_sim, neg_sim):
        # concatenate positive and negative similarities
        sim = torch.cat([pos_sim, neg_sim], dim=1)
        # compute softmax
        sim_softmax = torch.softmax(sim, dim=1)
        # return negative log likelihood of positive samples
        return -torch.log(sim_softmax[:, :pos_sim.size(1)]).mean()


class ModifiedBCELoss(nn.Module):
    def __init__(self):
        super(ModifiedBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pos_sim, neg_sim):
        # concatenate positive and negative similarities
        sim = torch.cat([pos_sim, neg_sim], dim=1)

        # create labels for positive (1) and negative (0) samples
        labels = torch.cat([
            torch.ones(pos_sim.size(0), dtype=torch.float32, device=pos_sim.device),
            torch.zeros(neg_sim.size(0), dtype=torch.float32, device=neg_sim.device)
        ])

        # compute BCE loss
        return self.bce(sim.squeeze(), labels)


class NormalizedLoss(nn.Module):
    def __init__(self):
        super(NormalizedLoss, self).__init__()

    def forward(self, pos_sim, neg_sim, pos_count, neg_count):
        pos_sim = pos_sim.to('cuda')
        neg_sim = neg_sim.to('cuda')
        pos_loss = -torch.log(torch.softmax(pos_sim, dim=1)).mean()
        neg_loss = -torch.log(torch.softmax(neg_sim, dim=1)).mean()
        # Assume pos_count and neg_count are the number of positive and negative samples respectively
        return pos_loss / pos_count + neg_loss / neg_count


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, pos_sim, neg_sim):
        pos_loss = torch.clamp(1 - pos_sim, min=0)
        neg_loss = torch.clamp(neg_sim + self.margin, min=0)
        loss = pos_loss.mean() + neg_loss.mean()
        return loss

