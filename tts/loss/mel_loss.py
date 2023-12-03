import torch
import torch.nn as nn


class MelLoss(nn.Module):
    def __init(self):
        super().__init()
        self.Loss = nn.L1Loss()

    def forward(self, target_mels, pred_mels):
        return self.Loss(target_mels, pred_mels)