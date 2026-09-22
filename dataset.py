import os
import torch
import numpy as np
import scipy.signal as signal
from torch.utils.data import Dataset, DataLoader
import torchaudio

try:
    import pywt
except ImportError:
    pywt = None

# Audio parameters matching published paper
SAMPLE_RATE = 2000
OLD_SAMPLE_RATE = 2000
CUTOFF_FREQ_LOW = 25
CUTOFF_FREQ_HIGH = 100

class WaveletTransform(torch.nn.Module):
    """
    Multi-scale Wavelet Denoising using Daubechies 4 (db4) with soft thresholding
    and inverse wavelet reconstruction. Retains original signal dimensions.
    """
    def __init__(self, wavelet='db4', level=4):
        super(WaveletTransform, self).__init__()
        self.wavelet = wavelet
        self.level = level

    def forward(self, x):
        if pywt is None:
            return x
        
        orig_shape = x.shape
        x_np = x.detach().cpu().squeeze().numpy()
        orig_len = len(x_np) if x_np.ndim == 1 else x_np.shape[-1]

        # Handle 1D signal
        if x_np.ndim == 1:
            coeffs = pywt.wavedec(x_np, self.wavelet, level=self.level)
            # Universal threshold (VisuShrink)
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            uthresh = sigma * np.sqrt(2.0 * np.log(max(orig_len, 2)))
            coeffs_thresh = [coeffs[0]] + [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs[1:]]
            reconstructed = pywt.waverec(coeffs_thresh, self.wavelet)
            if len(reconstructed) > orig_len:
                reconstructed = reconstructed[:orig_len]
            elif len(reconstructed) < orig_len:
                reconstructed = np.pad(reconstructed, (0, orig_len - len(reconstructed)))
            out = torch.tensor(reconstructed, dtype=x.dtype, device=x.device)
            return out.view(orig_shape)
        return x


class IIRFilter(torch.nn.Module):
    """
    Infinite Impulse Response (IIR) Butterworth Filter for smoothing
    and high-frequency noise suppression (Paper Section II-A).
    """
    def __init__(self, cutoff=100, fs=2000, order=4, filter_type='low'):
        super(IIRFilter, self).__init__()
        self.cutoff = cutoff
        self.fs = fs
        self.order = order
        self.filter_type = filter_type
        # Precompute Butterworth filter coefficients
        nyq = 0.5 * self.fs
        normal_cutoff = self.cutoff / nyq
        self.b, self.a = signal.butter(self.order, normal_cutoff, btype=self.filter_type)

    def forward(self, x):
        orig_shape = x.shape
        x_np = x.detach().cpu().squeeze().numpy()
        if x_np.ndim == 1:
            filtered = signal.lfilter(self.b, self.a, x_np)
            out = torch.tensor(filtered, dtype=x.dtype, device=x.device)
            return out.view(orig_shape)
        return x


mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=2048,
    hop_length=512,
    n_mels=128,
)

amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

wavelet_transform = WaveletTransform(wavelet='db4', level=4)
iir_filter = IIRFilter(cutoff=CUTOFF_FREQ_HIGH, fs=SAMPLE_RATE, order=4, filter_type='low')

audio_transforms = torch.nn.Sequential(
    wavelet_transform,
    iir_filter,
    mel_spectrogram,
    amplitude_to_db,
)


class HeartSoundDataset(Dataset):
    def __init__(self,
                 audio_dir,
                 segment_duration_sec=5,
                 transformation=None,
                 target_sample_rate=2000,
                 device="cpu"):
        self.audio_dir = audio_dir
        self.segment_duration_sec = segment_duration_sec
        self.device = device
        self.transformation = transformation.to(device) if transformation else None
        self.target_sample_rate = target_sample_rate
        self.segment_sample_length = int(self.segment_duration_sec * self.target_sample_rate)

        self.segment_index = []
        if os.path.exists(audio_dir):
            self.classes = [d for d in os.listdir(audio_dir) if os.path.isdir(os.path.join(audio_dir, d))]
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
            for cls in self.classes:
                class_path = os.path.join(audio_dir, cls)
                for fname in os.listdir(class_path):
                    if fname.endswith(".wav"):
                        file_path = os.path.join(class_path, fname)
                        try:
                            info = torchaudio.info(file_path)
                            num_segments = (info.num_frames) // self.segment_sample_length
                            for i in range(max(1, num_segments)):
                                self.segment_index.append((file_path, cls, i))
                        except Exception:
                            continue
        else:
            self.classes = []
            self.class_to_idx = {}

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
