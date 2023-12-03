from torch import nn
from torch.nn import functional as F
import torch
from torch.nn.utils import weight_norm, spectral_norm


class ScaleBlock(nn.Module):
    def __init__(self, use_spectral_norm):
        super().__init__()
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = F.leaky_relu(l(x))
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)

        return x, fmap


class MultyScaleDiscriminator(nn.Module):
    def __init__(self, **keys):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ScaleBlock(True),
            ScaleBlock(False),
            ScaleBlock(False),
        ])
        self.poolings = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, audio, pred_audio):
        prob_audio, history_audio = self.discriminators[0](audio)
        prob_pred, history_pred = self.discriminators[0](pred_audio)
        lst_prob_audio, lst_prob_pred = [prob_audio], [prob_pred]
        lst_history_audio, lst_history_pred = [history_audio], [history_pred]
        for discriminator, pooling in zip(self.discriminators[1:], self.poolings):
            prob_audio, history_audio = discriminator(pooling(audio))
            prob_pred, history_pred = discriminator(pooling(pred_audio))
            lst_prob_audio.append(prob_audio)
            lst_prob_pred.append(prob_pred)
            lst_history_audio.append(history_audio)
            lst_history_pred.append(history_pred)

        return lst_prob_audio, lst_prob_pred, lst_history_audio, lst_history_pred
