import torchaudio
import torch
import pywt
import scipy.signal as signal
import os
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import models
from torch import nn
import numpy as np

#Replace with path to split directories
AUDIO_DIR = '/final/heart_sound/train' 
AUDIO_DIR_VAL = '/final/heart_sound/val'

SAMPLE_RATE = 2000
NUM_FRAMES = 300000
OLD_SAMPLE_RATE = 2000
NEW_SAMPLE_RATE = 2000
CUTTOFF_FREQ_HIGH = 400
CUTTOFF_FREQ_LOW = 25
VOLUME = 0.5
NUM_FRAMES = 300000
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class WaveletTransform(nn.Module):
    def __init__(self, wavelet='db4', level=4):
        super(WaveletTransform, self).__init__()
        self.wavelet = wavelet
        self.level = level

    def forward(self, x):
        x_np = x.squeeze().cpu().numpy() 
        coeffs = pywt.wavedec(x_np, self.wavelet, level=self.level)
        coeffs_flat = np.hstack(coeffs)  
        coeffs_tensor = torch.tensor(coeffs_flat, dtype=torch.float32).unsqueeze(0) 
        return x


class IIRFilter(nn.Module):
    def __init__(self, cutoff=100, fs=2000, order=4, filter_type='low'):
        super(IIRFilter, self).__init__()
        self.cutoff = cutoff
        self.fs = fs
        self.order = order
        self.filter_type = filter_type

    def forward(self, x):
        b, a = signal.butter(self.order, self.cutoff / (0.5 * self.fs), btype=self.filter_type)
        x_np = x.squeeze().cpu().numpy()  
        filtered = signal.lfilter(b, a, x_np)
        filtered_tensor = torch.tensor(filtered, dtype=torch.float32).unsqueeze(0) 
        return x
    
mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=OLD_SAMPLE_RATE,
    n_fft=2048,
    hop_length=512,
    n_mels=128,
)

amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

class HeartSoundDataset(Dataset):
    def __init__(self,
                 audio_dir,
                 segment_duration_sec,
                 transformation=None,
                 target_sample_rate=22050,
                 device="cpu"):
        self.audio_dir = audio_dir
        self.segment_duration_sec = segment_duration_sec
        self.device = device
        self.transformation = transformation.to(device) if transformation else None
        self.target_sample_rate = target_sample_rate
        self.segment_sample_length = int(self.segment_duration_sec * self.target_sample_rate)

        self.classes = os.listdir(audio_dir)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.segment_index = [] 


        for cls in self.classes:
            class_path = os.path.join(audio_dir, cls)
            if not os.path.isdir(class_path):
                continue
            for fname in os.listdir(class_path):
                if fname.endswith(".wav"):
                    file_path = os.path.join(class_path, fname)
                    info = torchaudio.info(file_path)
                    orig_sample_rate = info.sample_rate
                    num_samples = info.num_frames

                    duration_sec = num_samples / orig_sample_rate
                    total_target_samples = int(duration_sec * self.target_sample_rate)
                    num_segments = total_target_samples // self.segment_sample_length

                    for i in range(num_segments):
                        self.segment_index.append((file_path, cls, i))

    def __len__(self):
        return len(self.segment_index)

    def __getitem__(self, idx):
        file_path, class_name, segment_idx = self.segment_index[idx]
        label = self.class_to_idx[class_name]

        waveform, orig_sample_rate = torchaudio.load(file_path)
        waveform = waveform.to(self.device)

        if orig_sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_sample_rate, self.target_sample_rate).to(self.device)
            waveform = resampler(waveform)

        waveform = self._mix_down_if_necessary(waveform)

        start = segment_idx * self.segment_sample_length
        end = start + self.segment_sample_length
        segment = waveform[:, start:end]

        if segment.shape[1] < self.segment_sample_length:
            segment = self._pad_if_necessary(segment)

        if self.transformation:
            segment = self.transformation(segment)

        return segment, label

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _pad_if_necessary(self, signal):
        signal_length = signal.shape[1]
        if signal_length < self.segment_sample_length:
            repeat_factor = (self.segment_sample_length + signal_length - 1) // signal_length
            signal = signal.repeat(1, repeat_factor)[:, :self.segment_sample_length]
        return signal


wavelet_transform = WaveletTransform(wavelet='db4', level=4)
iir_filter = IIRFilter(cutoff=100, fs=SAMPLE_RATE, order=4, filter_type='low')
audio_transforms = torch.nn.Sequential(
    wavelet_transform,
    iir_filter,
    mel_spectrogram,
    amplitude_to_db,
)

hsd = HeartSoundDataset(
    audio_dir=AUDIO_DIR,
    segment_duration_sec=5,
    transformation=audio_transforms,
    target_sample_rate=SAMPLE_RATE,
    device='cpu',
)

hsd_val = HeartSoundDataset(
    audio_dir=AUDIO_DIR_VAL,
    segment_duration_sec=5,
    transformation=audio_transforms,
    target_sample_rate=SAMPLE_RATE,
    device='cpu',
)

print(len(hsd))
print(len(hsd_val))
