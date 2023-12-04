# Mel2Wav

## Установка

Для установки для начала потребуется  установить библиотеки, скачать данные и модель
```shell
pip install -r ./requirements.txt
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 -o /dev/null
mkdir data
tar -xvf LJSpeech-1.1.tar.bz2 >> /dev/null
mv LJSpeech-1.1 data/LJSpeech-1.1
```
Модель лежит по ссылке - https://disk.yandex.ru/d/MT-ns08rraj6Gw


## Модель

Реализация основана на статье HiFiGAN с использованием hydra в качестве конфига.
https://arxiv.org/pdf/2010.05646.pdf


## Проверка на данных

Чтобы запустить тренировку модели на части LibriSpeech нужно изменить config, а именно часть, отвечающую за датасеты.
Выглядит она так
```yaml
batch_size: 25
batch_expand_size: 1
num_workers: 10

datasets:
  - _target_: tts.datasets.BufferDataset
    data_path: 'data/LJSpeech-1.1'
    slice_length: 8192
```
Все, что нужно подправить, это указать по какому пути лежит датасет.

Также можно протестировать модель на произвольном датасете, для этого нужно поменять конфиг данных для тренировки6 устроен он следующим образом:
```yaml
defaults:
  - arch: HiFiGAN
  # - augmentations: base_augmentations

name: test_config
n_gpu: 1
checkpoint: 
```
Для запуска нужно указать путь до модели с помощью checkpoint.

## Credits

This repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.

