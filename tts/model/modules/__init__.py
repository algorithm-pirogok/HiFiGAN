from tts.model.modules.generator import Generator
from tts.model.modules.period_discriminator import MultyPeriodDiscriminator
from tts.model.modules.scale_discriminator import MultyScaleDiscriminator

__all__ = [
    "Generator",
    "MultyScaleDiscriminator",
    "MultyPeriodDiscriminator",
]
