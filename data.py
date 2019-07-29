import numpy as np
import os

import torch
from torch.utils.data import Dataset, DataLoader

import hparams


class LJDataset(Dataset):

    def __init__(self, root_dir, train=True):
        """
        :param root_dir(string): Directory with all the spectrograms.
        :param train(bool): Type of dataset: train, test.
        """
        self.clear_data = os.path.join(root_dir, 'clear')
        self.noise_data = os.path.join(root_dir, 'noise')
        self.train = train
        self.names = self.collect_names()

    def __len__(self):
        return len(self.names)

    def collect_names(self):
        num_samples = len(os.listdir(self.clear_data))
        split = int(0.95*num_samples)
        if self.train:
            return os.listdir(self.clear_data)[:split]
        else: return os.listdir(self.clear_data)[split:]

    def __getitem__(self, idx):

        clear_spec = np.load(os.path.join(self.clear_data, self.names[idx]))
        noise_spec = np.load(os.path.join(self.noise_data, self.names[idx]))
        return clear_spec, noise_spec


def _pad_2d(x, max_len, b_pad=0):
    """
    Pad images to fit into max_len size
    """
    x = np.pad(x, [(b_pad, max_len - len(x) - b_pad), (0, 0)],
               mode="constant", constant_values=0)
    return x


def collate_fn(batch):
    """
    Create batch
    :param batch(tuple): list of x: (F, T)
    :return: Tuple of batches (B,C,T)
    """
    new_batch = []
    max_len = max(len(x) for x in batch)
    for (x, y) in batch:
        s = np.random.randint(0, x.size(1) - hparams.max_time_frames)
        x[:] = x[s:s + hparams.max_time_frames]
        y[:] = y[s:s + hparams.max_time_frames]
        new_batch.append((_pad_2d(x, max_len), _pad_2d(y, max_len)))

    return torch.tensor(new_batch)


if __name__ == '__main__':

    dataset = LJDataset('./datasets/ljspeech')
    data_loader = DataLoader(dataset=dataset, batch_size=1)
    next_batch = next(iter(data_loader))
    print(next_batch.size())