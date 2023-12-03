from torch import nn
from torch.nn import functional as F
import torch

from tts.model.modules import Generator, MultyPeriodDiscriminator, MultyScaleDiscriminator
from tts.base.base_model import BaseModel


class HiFiGAN(BaseModel):
    def __init__(self, generator_params, scale_discriminator_params, period_discriminator_params, **keys):
        super(HiFiGAN, self).__init__()
        self.generator = Generator(**generator_params)
        self.scale_discriminator = MultyScaleDiscriminator(**scale_discriminator_params)
        self.period_discriminator = MultyPeriodDiscriminator(**period_discriminator_params)

    def forward(self, mels, target_audio, **args):
        pass
