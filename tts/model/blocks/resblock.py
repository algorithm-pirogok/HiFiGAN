import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm


class ResBlock(nn.Module):
    def __init__(self, channels, kernel, dilations):
        super().__init__()
        self.first_convs = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel,
                        1,
                        dilation=dilations[0],
                        padding=(kernel - 1) * dilations[0] // 2,
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel,
                        1,
                        dilation=dilations[1],
                        padding=(kernel - 1) * dilations[1] // 2,
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel,
                        1,
                        dilation=dilations[2],
                        padding=(kernel - 1) * dilations[2] // 2,
                    )
                ),
            ]
        )
        self.second_convs = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel,
                        1,
                        dilation=1,
                        padding=(kernel - 1) // 2,
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel,
                        1,
                        dilation=1,
                        padding=(kernel - 1) // 2,
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel,
                        1,
                        dilation=1,
                        padding=(kernel - 1) // 2,
                    )
                ),
            ]
        )

    def forward(self, x):
        for first_conv, second_conv in zip(self.first_convs, self.second_convs):
            x_tmp = first_conv(F.leaky_relu(x))
            x_tmp = second_conv(F.leaky_relu(x_tmp))
            x = x + x_tmp
        return x
