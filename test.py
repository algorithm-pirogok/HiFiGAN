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
from tqdm import tqdm
import pyloudnorm as pyln

import tts.model as module_model
from tts.trainer import Trainer
from tts.utils import ROOT_PATH, get_logger
from tts.utils.object_loading import get_dataloaders

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


@hydra.main(config_path='tts/configs', config_name='test_config')
def main(clf):


    # define cpu or gpu if possible
    device = "cpu"

    # setup data_loader instances
    # dataloaders = get_dataloaders(clf)

    # build model architecture
    model = instantiate(clf["arch"])
    print(clf.checkpoint)
    checkpoint = torch.load(clf.checkpoint, map_location=device)
    print(checkpoint)
    state_dict = checkpoint["state_dict"]
    if clf["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()
    exit()
    results = []
    metrcics = defaultdict(list)


    norm_wav = pyln.Meter(clf["preprocessing"]["sr"])

    with torch.no_grad():
        for dataset in dataloaders.keys():
            for batch_num, batch in enumerate(tqdm(dataloaders[dataset])):
                batch = Trainer.move_batch_to_device(batch, device)
                output = model(**batch)
                if type(output) is dict:
                    batch.update(output)
                else:
                    raise Exception("change type of model")
                for ind in range(len(batch['snr'])):
                    for mode in ('short', 'middle', 'long'):
                        batch[mode][ind] = torch.tensor(pyln.normalize.loudness(
                            batch[mode][ind].cpu().numpy(),
                            norm_wav.integrated_loudness(batch[mode][ind].cpu().numpy()),
                            -23.0
                        )).to(device)
                    curr_batch = {key: batch[key][ind] for key in batch.keys()}
                    metrcics["si_sdr"].append(si_sdr(**curr_batch).item())
                    metrcics["comb_si_sdr"].append(comb_si_sdr(**curr_batch).item())
                    metrcics["pesq"].append(pesq(**curr_batch).item())
                    metrcics["comb_pesq"].append(comb_pesq(**curr_batch).item())

                print("Iteration:", batch_num)
                for key, value in metrcics.items():
                    print(f"{key}: {np.mean(value)}")

    final_dict = {}
    for key, value in metrcics.items():
        final_dict[key] = np.mean(value)
        print(f"{key}: {np.mean(value)}")

    with open(clf.out_file, "w") as f:
        json.dump(final_dict, f, indent=2)

    '''
            logger.info(f"butch_num {batch_num}, len_of_object {len(metrcics['text_argmax'])}")

            for key, history in metrcics.items():
                wer, cer = zip(*history)
                wer = np.mean(wer)
                cer = np.mean(cer)
                logger.info(f'{mode} {key}_WER = {wer}')
                logger.info(f'{mode} {key}_CER = {cer}')

            with Path(out_file).open("w") as f:
                json.dump(models, f, indent=2)'''


if __name__ == "__main__":
    main()
