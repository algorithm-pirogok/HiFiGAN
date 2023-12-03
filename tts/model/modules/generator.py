import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from tts.model.blocks.resblock import ResBlock


class Generator(nn.Module):
    def __init__(
        self,
        in_channels,
        upsample_channel,
        conv_kernels,
        conv_strides,
        res_kernels,
        res_dilations,
    ):
        super().__init__()
        self.input_convolution = weight_norm(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=upsample_channel,
                kernel_size=7,
                stride=1,
                padding=3,
            )
        )
        self.resnet_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.LeakyReLU(),
                        weight_norm(
                            nn.ConvTranspose1d(
                                in_channels=upsample_channel // (2**num),
                                out_channels=upsample_channel // (2 ** (num + 1)),
                                kernel_size=kernel,
                                stride=stride,
                                padding=(kernel - stride) // 2,
                            )
                        ),
                        nn.ModuleList(
                            [
                                ResBlock(
                                    channels=upsample_channel // (2 ** (num + 1)),
                                    kernel=resnet_kernel,
                                    dilations=res_dilations,
                                )
                                for resnet_kernel in res_kernels
                            ]
                        ),
                    ]
                )
                for num, (kernel, stride) in enumerate(zip(conv_kernels, conv_strides))
            ]
        )

        self.output_convolution = weight_norm(
            nn.Conv1d(
                in_channels=upsample_channel // (2 ** len(self.resnet_blocks)),
                out_channels=1,
                kernel_size=7,
                stride=1,
                padding=3,
            )
        )

    def forward(self, mels):
        ans = self.input_convolution(mels)
        for relu, conv, resblocks in self.resnet_blocks:
            ans = F.leaky_relu(conv(ans))
            curr = None
            for block in resblocks:
                curr = block(ans) + curr if curr is not None else block(ans)
            ans = curr / len(resblocks)
        ans = F.leaky_relu(ans)
        ans = self.output_convolution(ans)
        return torch.tanh(ans)
