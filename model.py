import torch
from torch import nn

from copy import deepcopy

Conv = nn.Conv2d

import argparse
parser = argparse.ArgumentParser(argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--torchaudio_path', type=str, default='torchaudio')
args = parser.parse_args()

import sys
sys.path.append(args.torchaudio_path)
import functional

import hparams

class DeepGL(nn.Module):

    def __init__(self, num_blocks=20):
        """
        :param num_blocks(int): Number of Deep Griffin Lim blocks.
        :param n_fft(int): FFT window size
        """
        super().__init__()

        self.num_blocks = num_blocks
        self.block = DeGLI_block()

    def forward(self, x):
        amplitude = deepcopy(x)
        z = None
        for i in range(self.num_blocks):
            z, x[:] = self.block(x, amplitude)

        return z, x


class DeGLI_block(nn.Module):
    """
    Deep Griffin Lim block
    """

    def __init__(self):
        super().__init__()

        self.pa = lambda amplitude, x: amplitude * x / torch.norm(x)

        self.dnn1 = nn.Sequential(
            Conv(in_channels=6, out_channels=32, kernel_size=11, stride=1, padding=5),
            nn.GLU(dim=1)
        )

        self.dnn2 = nn.Sequential(
            Conv(in_channels=16, out_channels=32, kernel_size=(7, 3), stride=1, padding=(3, 1)),
            nn.GLU(dim=1),
            Conv(in_channels=16, out_channels=32, kernel_size=(7, 3), stride=1, padding=(3, 1)),
            nn.GLU(dim=1)
        )

        self.dnn3 = nn.Sequential(
            Conv(in_channels=16, out_channels=32, kernel_size=(7, 3), stride=1, padding=(3, 1)),
            nn.GLU(dim=1),
            Conv(in_channels=16, out_channels=2, kernel_size=(7, 3), stride=1, padding=(3, 1)),
        )

    def forward(self, x, amplitude):
        x = x.permute(0, 2, 3, 1)
        y = self.pa(amplitude.permute(0, 2, 3, 1), x)

        z = functional.istft(y, hparams.fft_size)
        z = torch.stft(z, hparams.fft_size)

        input = torch.cat((x, y, z), dim=3)
        z = z.permute(0, 3, 1, 2)
        del x, y
        input = input.permute(0, 3, 1, 2)
        input = self.dnn1(input)

        return z, z - self.dnn3(self.dnn2(input) + input)


if __name__ == '__main__':
    device = torch.cuda()
    model = DeepGL(3)
    model.to(device)

    a = torch.rand(4, 2, 513, 1030).to(device)

    z, out = model(a)

