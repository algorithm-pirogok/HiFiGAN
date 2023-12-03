import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm


class PeriodBlock(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, *args, **kwargs):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, (kernel_size, 1),
                                  (stride, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(32, 128, (kernel_size, 1),
                                  (stride, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(128, 512, (kernel_size, 1),
                                  (stride, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(512, 1024, (kernel_size, 1),
                                  (stride, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(1024, 1024, (kernel_size, 1),
                                  1, padding=(2, 0))),
        ])
        self.conv_post = weight_norm(nn.Conv2d(1024, 1, (3, 1),
                                               1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)

        return x, fmap


class MultyPeriodDiscriminator(nn.Module):
    def __init__(self, **keys):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PeriodBlock(2),
            PeriodBlock(3),
            PeriodBlock(5),
            PeriodBlock(7),
            PeriodBlock(11),
        ])

    def forward(self, audio, pred_audio):
        lst_prob_audio, lst_history_audio = [], []
        lst_prob_pred, lst_history_pred = [], []
        for discriminator in self.discriminators:
            prob_audio, history_audio = discriminator(audio)
            prob_pred, history_pred = discriminator(pred_audio)
            lst_prob_audio.append(prob_audio)
            lst_history_audio.append(history_audio)
            lst_prob_pred.append(prob_pred)
            lst_history_pred.append(history_pred)
        return lst_prob_audio, lst_prob_pred, lst_history_audio, lst_history_pred
