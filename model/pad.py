import torch
import torch.nn as nn


class PadForMelspectrogram(nn.Module):
    def __init__(self, hp):
        super(PadForMelspectrogram, self).__init__()
        self.pad = (int((hp.audio.filter_length - hp.audio.hop_length) / 2), int((hp.audio.filter_length - hp.audio.hop_length) / 2))

    def forward(self, x):
        x = torch.nn.functional.pad(x,
                                self.pad,
                                mode='reflect')
        return x