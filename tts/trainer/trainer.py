from itertools import chain
from pathlib import Path

import numpy as np
import PIL
import torch
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from tts.base import BaseTrainer
from tts.datasets.MelSpectrogram import MelSpectrogram
from tts.logger.utils import plot_spectrogram_to_buf
from tts.utils import ROOT_PATH, MetricTracker, inf_loop
from tts.utils.util import MelSpectrogramConfig


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
        self,
        model,
        criterion,
        metrics_train,
        metrics_test,
        generator_optimizer,
        discriminator_optimizer,
        config,
        device,
        log_step,
        dataloader,
        generator_scheduler=None,
        discriminator_scheduler=None,
        len_epoch=None,
        skip_oom=True,
    ):
        super().__init__(
            model,
            criterion,
            metrics_train,
            metrics_test,
            generator_optimizer,
            discriminator_optimizer,
            config,
            device,
        )
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloader = dataloader
        self.generator_scheduler = generator_scheduler
        self.discriminator_scheduler = discriminator_scheduler
        self.log_step = log_step
        self.wav_to_mel = MelSpectrogram(MelSpectrogramConfig())
        self.metrics = [
            "general_loss",
            "discriminator_loss",
            "mel_loss",
            "generator_loss",
            "feature_loss",
            "scale_loss",
            "period_loss",
        ]
        self.train_metrics = MetricTracker(
            "discriminator grad norm",
            "generator grad norm",
            *self.metrics,
            writer=self.writer,
        )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        names = ["target_audio", "target_mels"]
        for tensor_for_gpu in names:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self, params=None):
        if params is None:
            params = self.model.parameters()
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(params, self.config["trainer"]["grad_norm_clip"])

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)

        for batch_idx, batch in enumerate(
            tqdm(self.train_dataloader, desc="train", total=self.len_epoch / 15)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    batch_idx=batch_idx * 15,
                    is_train=True,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            print(batch_idx, self.log_step)
            if batch_idx * 15 % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx * 15)
                self.logger.debug(
                    f"Train Epoch: {epoch} Generator Loss: {batch['general_loss']} "
                    f"Discriminator Loss: {batch['discriminator_loss']}"
                )
                self.writer.add_scalar(
                    "generator learning rate",
                    self.generator_optimizer.param_groups[0]["lr"],
                )
                self.writer.add_scalar(
                    "discriminator learning rate",
                    self.discriminator_optimizer.param_groups[0]["lr"],
                )
                self._log_scalars(self.train_metrics)
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
                self._log(
                    batch["target_mels"],
                    batch["pred_mels"],
                    batch["target_audio"],
                    batch["pred_audio"],
                )
                if batch_idx * 15 >= self.len_epoch:
                    return last_train_metrics

        log = last_train_metrics

        return log

    def process_batch(
        self, batch, batch_idx: int, is_train: bool, metrics: MetricTracker
    ):
        batch = self.move_batch_to_device(batch, self.device)

        batch["pred_audio"] = self.model.generator(batch["target_mels"])
        batch["pred_mels"] = self.wav_to_mel(batch["pred_audio"].squeeze(1))

        real_scale, pred_scale, _, _ = self.model.scale_discriminator(
            batch["target_audio"], batch["pred_audio"].detach()
        )
        real_period, pred_period, _, _ = self.model.period_discriminator(
            batch["target_audio"], batch["pred_audio"].detach()
        )
        batch["scale_loss"] = self.criterion.discriminator_loss(real_scale, pred_scale)
        batch["period_loss"] = self.criterion.discriminator_loss(
            real_period, pred_period
        )

        batch["discriminator_loss"] = batch["scale_loss"] + batch["period_loss"]

        if is_train:
            self.discriminator_optimizer.zero_grad()
            batch["discriminator_loss"].backward()
            self._clip_grad_norm(self.model.scale_discriminator.parameters())
            self._clip_grad_norm(self.model.period_discriminator.parameters())
            self.discriminator_optimizer.step()
            self.discriminator_scheduler.step()
            metrics.update(
                "discriminator grad norm",
                self.get_grad_norm(
                    chain(
                        self.model.scale_discriminator.parameters(),
                        self.model.period_discriminator.parameters(),
                    )
                ),
            )

        batch["mel_loss"] = self.criterion.mel_loss(
            batch["target_mels"], batch["pred_mels"]
        )

        (
            _,
            pred_scale,
            scale_real_history,
            scale_pred_history,
        ) = self.model.scale_discriminator(batch["target_audio"], batch["pred_audio"])
        (
            _,
            pred_period,
            real_period_history,
            pred_period_history,
        ) = self.model.scale_discriminator(batch["target_audio"], batch["pred_audio"])

        batch["feature_loss"] = self.criterion.feature_loss(
            scale_real_history, scale_pred_history
        ) + self.criterion.feature_loss(real_period_history, pred_period_history)

        batch["generator_loss"] = self.criterion.generator_loss(
            pred_scale
        ) + self.criterion.generator_loss(pred_period)

        batch["general_loss"] = (
            batch["mel_loss"] + batch["feature_loss"] + batch["generator_loss"]
        )

        if is_train:
            self.generator_optimizer.zero_grad()
            batch["general_loss"].backward()
            self._clip_grad_norm(self.model.generator.parameters())
            self.generator_optimizer.step()
            self.generator_scheduler.step()
            metrics.update(
                "generator grad norm",
                self.get_grad_norm(self.model.generator.parameters()),
            )

        for loss_name in self.metrics:
            metrics.update(loss_name, batch[loss_name].item())
        return batch

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log(self, true_mels, pred_mels, true_audio, pred_audio):
        idx = np.random.choice(np.arange(len(true_mels)))
        img_true = PIL.Image.open(
            plot_spectrogram_to_buf(true_mels[idx].detach().cpu().numpy().T)
        )
        img_pred = PIL.Image.open(
            plot_spectrogram_to_buf(pred_mels[idx].detach().cpu().numpy().T)
        )
        self.writer.add_image("Target mel", ToTensor()(img_true))
        self.writer.add_image("Prediction mel", ToTensor()(img_pred))
        self.writer.add_audio("Target audio", true_audio.squeeze(), sample_rate=22050)
        self.writer.add_audio(
            "Prediction audio", pred_audio.squeeze(), sample_rate=22050
        )

    @torch.no_grad()
    def get_grad_norm(self, parameters, norm_type=2):
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
