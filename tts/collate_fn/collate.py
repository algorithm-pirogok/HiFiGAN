import logging
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from tts.collate_fn.functions import reprocess_tensor
from tts.datasets.MelSpectrogram import MelSpectrogram
from tts.utils.util import MelSpectrogramConfig


class collate_fn:
    def __init__(self):
        config = MelSpectrogramConfig()
        self.wav_to_mel = MelSpectrogram(config).to('cpu')

    def __call__(self, dataset_items: List[dict]):
        """
        Collate and pad fields in dataset items
        """

        audio_length = torch.tensor(
            [dataset["target_audio"].shape[-1] for dataset in dataset_items]
        )
        target_audio = torch.vstack([dataset["target_audio"] for dataset in dataset_items])
        # target_audio = torch.zeros((len(dataset_items), max(audio_length)))
        # for num, (item, len_audio) in enumerate(zip(dataset_items, audio_length)):
        #    target_audio[num, :len_audio] = torch.tensor(item["target_audio"])

        target_mels = self.wav_to_mel(target_audio)

        return {
            "target_audio": torch.unsqueeze(target_audio, 1),
            "audio_length": audio_length,
            "target_mels": target_mels,
        }
