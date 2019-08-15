"""
Hyperparameters for audio and spectogram preprocessing
"""

sample_rate = 16000
fmin = 125
fmax = 7600
fft_size = 1024
hop_size = 256
win_length = None
min_level_db = -100
ref_level_db = 20
rescaling_max = 0.999
snr_max = 0
snr_min = -6
max_time_steps = 32000
max_steps = max_time_steps - max_time_steps % hop_size
max_time_frames = max_steps // hop_size
