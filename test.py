import argparse
from collections import defaultdict
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np
import hydra
from hydra.utils import instantiate
import torch
import torchaudio
from tqdm import tqdm
import pyloudnorm as pyln

import tts.model as module_model
from tts.datasets.MelSpectrogram import MelSpectrogram
from tts.trainer import Trainer
from tts.utils import ROOT_PATH, MelSpectrogramConfig
from tts.utils.object_loading import get_dataloaders

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


@hydra.main(config_path='tts/configs', config_name='test_config')
def main(clf):

    # define cpu or gpu if possible
    device = "cpu"

    model = instantiate(clf["arch"])
    checkpoint = torch.load(clf.checkpoint, map_location=device)
    state_dict = checkpoint["state_dict"]
    if clf["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        config = MelSpectrogramConfig()
        wav_to_mel = MelSpectrogram(config)
        directory = ROOT_PATH / "data" / "test"
        directory_save = ROOT_PATH / "data" / "predictions"
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                target_audio, sample_rate = torchaudio.load(filepath)
                target_mel = wav_to_mel(target_audio)
                pred_audio = model.generator(target_mel).squeeze(0)
                torchaudio.save(directory_save / filename, pred_audio, sample_rate)

if __name__ == "__main__":
    main()
