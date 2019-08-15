import argparse
import os

from scipy import signal
from scipy.io import wavfile

import librosa

import numpy as np

from tqdm import tqdm

import hparams


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str, default='/workspace/raid/data/anagapetyan/DeepGL/data/STC/Train/train/auto', help='Dateset path')
parser.add_argument('--prep_path', type=str, default='/workspace/raid/data/anagapetyan/DeepGL/data/stc_linear', help='Path to save linear spectrogram')


def complex2real(x):
    """
    Convert complex-valued spectrogram to real valued with two real dimensions for real and imaginary part separately
    :param x(numpy array): complex-valued spectorgram
    :return:
    """
    return np.stack((x.real, x.imag), axis=2)


def _add_noise(audio, snr):
    """
    Add complex gaussian noise to signal with given SNR.
    :param audio(np.array):
    :param snr(float): sound-noise-ratio
    :return: audio with added noise
    """

    audio_mean = np.mean(audio**2)
    audio_mean_db = 10 * np.log10(audio_mean)

    noise_mean_db = snr - audio_mean_db
    noise_mean = 10 ** (noise_mean_db/10)

    return audio + np.random.normal(0, np.sqrt(noise_mean), len(audio))


def prepare_dirs(args):
    if not os.path.isdir(args.data_path):
        raise Exception("Invalid path. No such directory")

    if not os.path.isdir(args.prep_path):
        os.makedirs(args.prep_path)
        os.makedirs((os.path.join(args.prep_path, 'clear')))
        os.makedirs((os.path.join(args.prep_path, 'noise')))


def preprocess(args):
    """
    Add noise to audio and save its noisy and clear complex-valued spectrogram as two channel real matrices.
    """
    clear_spec = os.path.join(args.prep_path, 'clear')
    noise_spec = os.path.join(args.prep_path, 'noise')
    for file in tqdm(os.listdir(args.data_path)):
        if file.endswith('wav'):
            sr, audio = wavfile.read(os.path.join(args.data_path, file))

            snr = np.random.rand(1) * (hparams.snr_min - hparams.snr_max)
            noisy_audio = _add_noise(audio, snr)

            if hparams.rescaling_max:
                audio = audio / np.abs(audio).max() * hparams.rescaling_max

            spec = librosa.core.stft(np.float32(audio), hop_length=hparams.hop_size, n_fft=hparams.fft_size,
                                     win_length=hparams.win_length)

            if spec.shape[1] <= hparams.max_time_frames:
                continue
            #fr, times, spec = signal.spectrogram(audio, fs=sr)

            #min_level = np.exp(hparams.min_level_db / 20 * np.log(10))
            #spec = 20 * np.log10(np.maximum(min_level, (np.abs(spec)))) - hparams.ref_level_db
            #spec = np.clip((spec - hparams.min_level_db) / -hparams.min_level_db, 0, 1)
            filename = file[:-3] + 'npy'
            spec = complex2real(spec)
            np.save(os.path.join(clear_spec, filename), spec, allow_pickle=False)

            #noisy_audio = _add_noise(audio, snr)

            if hparams.rescaling_max:
                noisy_audio = audio / np.abs(audio).max() * hparams.rescaling_max

            spec = librosa.core.stft(np.float32(noisy_audio), hop_length=hparams.hop_size, n_fft=hparams.fft_size,
                                     win_length=hparams.win_length)
            #spec = 20 * np.log10(np.maximum(min_level, (np.abs(spec)))) - hparams.ref_level_db
            #spec = np.clip((spec - hparams.min_level_db) / -hparams.min_level_db, 0, 1)
            spec = complex2real(spec)
            np.save(os.path.join(noise_spec, filename), spec, allow_pickle=False)


if __name__ == '__main__':
    args = parser.parse_args()

    prepare_dirs(args)

    preprocess(args)
