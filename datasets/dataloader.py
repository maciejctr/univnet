import torch
import random
import torchaudio
import numpy as np
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

from utils.utils import read_wav_np
from utils.stft import TacotronSTFT


def create_dataloader(hp, args, train):
    if train:
        dataset = MelFromDisk(hp, Path(hp.data.data_dir), "train.tsv", args, train)
        return DataLoader(dataset=dataset, batch_size=hp.train.batch_size, shuffle=False,
                          num_workers=hp.train.num_workers, pin_memory=True, drop_last=True)

    else:
        dataset = MelFromDisk(hp, Path(hp.data.data_dir), "validation.tsv", args, train)
        return DataLoader(dataset=dataset, batch_size=1, shuffle=False,
            num_workers=hp.train.num_workers, pin_memory=True, drop_last=False)


class MelFromDisk(Dataset):
    def __init__(self, hp, data_dir, metadata, args, train):
        random.seed(hp.train.seed)
        self.hp = hp
        self.args = args
        self.train = train

        self.mel_segment_length = hp.audio.segment_length // hp.audio.hop_length
        self.shuffle = hp.train.spk_balanced
        self.data_dir = data_dir
        with open(Path(data_dir / metadata), 'r') as file:
            lines = file.readlines()
            self.audio_files= [Path(data_dir / line.split('\t')[0]) for line in lines]

        self.stft = torchaudio.transforms.MelSpectrogram(
                                sample_rate=hp.audio.sampling_rate,
                                n_fft=hp.audio.filter_length,
                                hop_length=hp.audio.hop_length,
                                n_mels=hp.audio.n_mel_channels,
                                center=False,
                                power=1.0,
                                normalized=True,
                                )

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        return self.my_getitem(idx)

    def my_getitem(self, idx):
        wavpath= self.audio_files[idx]
        sr, audio = read_wav_np(wavpath)

        if len(audio) < self.hp.audio.segment_length + self.hp.audio.pad_short:
            audio = np.pad(audio, (0, self.hp.audio.segment_length + self.hp.audio.pad_short - len(audio)), \
                    mode='constant', constant_values=0.0)

        audio = torch.from_numpy(audio).unsqueeze(0)
        mel = self.get_mel(wavpath)

        if self.train:
            max_mel_start = mel.size(1) - self.mel_segment_length -1
            mel_start = random.randint(0, max_mel_start)
            mel_end = mel_start + self.mel_segment_length
            mel = mel[:, mel_start:mel_end]

            audio_start = mel_start * self.hp.audio.hop_length
            audio_len = self.hp.audio.segment_length
            audio = audio[:, audio_start:audio_start + audio_len]

        return mel, audio

    def get_mel(self, wavpath):
        melpath = wavpath.with_suffix('.mel')
        try:
            mel = torch.load(melpath, map_location='cpu')
            assert mel.size(0) == self.hp.audio.n_mel_channels, \
                'Mel dimension mismatch: expected %d, got %d' % \
                (self.hp.audio.n_mel_channels, mel.size(0))

        except (FileNotFoundError, RuntimeError, TypeError, AssertionError):
            sr, wav = read_wav_np(wavpath)
            assert sr == self.hp.audio.sampling_rate, \
                'sample mismatch: expected %d, got %d at %s' % (self.hp.audio.sampling_rate, sr, wavpath)

            if len(wav) < self.hp.audio.segment_length + self.hp.audio.pad_short:
                wav = np.pad(wav, (0, self.hp.audio.segment_length + self.hp.audio.pad_short - len(wav)), \
                             mode='constant', constant_values=0.0)

            wav = torch.from_numpy(wav).unsqueeze(0)
            mel = self.stft(wav)

            mel = mel.squeeze(0)

            torch.save(mel, melpath)

        return mel

    def load_metadata(self, path, split="|"):
        metadata = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                stripped = line.strip().split(split)
                metadata.append(stripped)

        return metadata