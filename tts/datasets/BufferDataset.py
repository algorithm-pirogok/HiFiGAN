import os
import time

import numpy as np
import torchaudio
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pyworld as pw
import scipy.interpolate as interpolate

from tts.text import text_to_sequence
from tts.utils.util import ROOT_PATH
from tts.datasets.MelSpectrogram import MelSpectrogram


class BufferDataset(Dataset):
    def __init__(self, slice_length, *args, **kwargs):
        self.slice_length = slice_length
        self.buffer = self._get_buffer(*args, **kwargs)
        self.length_dataset = len(self.buffer)

    def _get_buffer(self, data_path):
        buffer = list()
        start = time.perf_counter()
        mn = 10000000
        for root, dirs, files in os.walk(ROOT_PATH / data_path / "wavs"):
            for file in files:
                path_file = os.path.join(root, file)
                audio, sample_rate = torchaudio.load(path_file) # 8192
                if audio.shape[1] < mn:
                    mn = audio.shape[1]
                buffer.append({"target_audio": audio[0]})
        end = time.perf_counter()
        print("cost {:.2f}s to load all data into buffer.".format(end - start))
        return buffer

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        elem = self.buffer[idx % 1]
        start_index = torch.randint(0, elem["target_audio"].shape[-1] - self.slice_length + 1, (1, ))
        elem["target_audio"] = elem["target_audio"][start_index:start_index+self.slice_length]

        return elem

    def _create_buffer(self):
        pass
