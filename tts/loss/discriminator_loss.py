import torch
import torch.nn as nn


class DiscriminatorLoss(nn.Module):
    def __init(self):
        super().__init()

    def forward(self, target_prob, pred_prob):
        loss =  0
        for tg, pr in zip(target_prob, pred_prob):
            loss = loss + torch.mean((tg-1)**2)+torch.mean(pr**2)
        return loss
