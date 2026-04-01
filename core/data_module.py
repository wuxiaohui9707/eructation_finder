import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import json
import csv
import torchaudio
import numpy as np
import torch
import random
import torch.nn.functional as F
import csv
from scipy import signal

class AudioDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for Audio Spectrogram Transformer"""
    
    def __init__(self, train_json_file=None, val_json_file=None, test_json_file=None, audio_conf=None, 
                label_csv=None, batch_size=32, num_workers=4, balanced_sampling=False,
                eval_audio_conf=None):
        super().__init__()
        self.save_hyperparameters()
        
        self.train_json_file = train_json_file
        self.val_json_file = val_json_file
        self.test_json_file = test_json_file
        self.audio_conf = audio_conf
        self.eval_audio_conf = eval_audio_conf if eval_audio_conf else audio_conf
        self.label_csv = label_csv
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.balanced_sampling = balanced_sampling
        
        # Initialize dataset parameters
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        self.skip_norm = self.audio_conf.get('skip_norm', False)
        self.noise = self.audio_conf.get('noise', False)
        
        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        """Load datasets from separate json files based on the stage."""
        if stage == "fit" or stage is None:
            if self.train_json_file:
                with open(self.train_json_file, 'r') as fp:
                    train_data = json.load(fp)['data']
                self.train_dataset = AudioDataset(train_data, self.audio_conf, self.label_csv)
            
            if self.val_json_file:
                with open(self.val_json_file, 'r') as fp:
                    val_data = json.load(fp)['data']
                self.val_dataset = AudioDataset(val_data, self.audio_conf, self.label_csv)
        
        if stage == "test" or stage is None:
            if self.test_json_file:
                with open(self.test_json_file, 'r') as fp:
                    test_data = json.load(fp)['data']
                self.test_dataset = AudioDataset(test_data, self.audio_conf, self.label_csv)

    def train_dataloader(self):
        # if self.balanced_sampling:
        #     # Get weights for all samples
        #     weights = []
        #     for i in range(len(self.train_dataset)):
        #         _, _, weight = self.train_dataset[i]
        #         weights.append(weight)
        #     sampler = torch.utils.data.WeightedRandomSampler(
        #         weights=weights,
        #         num_samples=len(weights),
        #         replacement=True
        #     )
        #     return DataLoader(
        #         self.train_dataset,
        #         batch_size=self.batch_size,
        #         num_workers=self.num_workers,
        #         sampler=sampler,
        #         pin_memory=True
        #     )
        # else:
        if self.train_dataset is None:
            raise ValueError("Train dataset is not loaded. Please provide train_json_file.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            raise ValueError("Validation dataset is not loaded. Please provide val_json_file.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            raise ValueError("Test dataset is not loaded. Please provide test_json_file.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )

def resample_audio(audio_path, target_sample_rate=16000, resample=False):
    """
    Resample audio to target sample rate
    Args: 
        audio_path: str, path to audio file
        target_sample_rate: int, target sample rate
        resample: bool, whether to resample audio
    Returns:
        waveform: torch.Tensor, resampled waveform
        sample_rate: int, sample rate of the waveform
    """

    waveform, original_sample_rate = torchaudio.load(audio_path)
    if resample:
        resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
        return waveform, target_sample_rate
    return waveform, original_sample_rate

def low_pass_filter(waveform_np, sample_rate, cutoff=1024):
    """
    Apply a low-pass filter to an audio signal.
    """
    nyquist = 0.5 * sample_rate
    cutoff_freq = cutoff / nyquist
    
    # Butterworth filter with order 5
    b, a = signal.butter(5, cutoff_freq, btype='low', analog=False)
    
    filtered_waveform_np = signal.filtfilt(b, a, waveform_np)
    filtered_waveform_np = filtered_waveform_np.astype(np.float32)
    
    return filtered_waveform_np

def get_custom_mel_filterbank(sample_rate, n_fft, n_mels, mode='uniform', split_freq=1000):
    """
    Generate custom mel filterbank with different frequency division modes.
    
    Args:
        sample_rate: Audio sample rate
        n_fft: FFT size
        n_mels: Number of mel bins
        mode: 'uniform' for standard mel-scale, 'split_1khz' for equal bins above/below split_freq
        split_freq: Frequency to split at (Hz), default 1000
    
    Returns:
        mel_fb: Mel filterbank tensor of shape (n_mels, n_fft // 2 + 1)
    """
    if mode == 'uniform':
        # Use standard mel filterbank
        mel_fb = torchaudio.functional.melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=0.0,
            f_max=sample_rate / 2.0,
            n_mels=n_mels,
            sample_rate=sample_rate,
            norm='slaney',
            mel_scale='htk'
        )
        return mel_fb.transpose(0, 1)
    elif mode == 'split_1khz':
        # Custom split at split_freq with equal bins above/below
        nyquist = sample_rate / 2.0
        n_freqs = n_fft // 2 + 1
        
        # Split n_mels in half
        half_mels = n_mels // 2
        
        # Create frequency points for lower half (0 to split_freq)
        lower_mels = torchaudio.functional.melscale_fbanks(
            n_freqs=n_freqs,
            f_min=0.0,
            f_max=split_freq,
            n_mels=half_mels,
            sample_rate=sample_rate,
            norm='slaney',
            mel_scale='htk'
        )
        
        # Create frequency points for upper half (split_freq to Nyquist)
        upper_mels = torchaudio.functional.melscale_fbanks(
            n_freqs=n_freqs,
            f_min=split_freq,
            f_max=nyquist,
            n_mels=n_mels - half_mels,  # Handle odd n_mels
            sample_rate=sample_rate,
            norm='slaney',
            mel_scale='htk'
        )
        
        # Concatenate lower and upper filterbanks
        # torchaudio melscale returns (n_freqs, n_mels)
        # We want to concatenate along the mel dimension (dim 1)
        mel_fb = torch.cat([lower_mels, upper_mels], dim=1)
        
        # Return (n_mels, n_freqs)
        return mel_fb.transpose(0, 1)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'uniform' or 'split_1khz'.")

class AudioDataset(torch.utils.data.Dataset):
    """Dataset for audio spectrogram processing"""
    
    def __init__(self, data, audio_conf, label_csv=None, balanced_sampling=False):
        self.data = data
        self.audio_conf = audio_conf
        self.label_csv = label_csv
        self.balanced_sampling = balanced_sampling
        
        self.index_dict = self._make_index_dict(label_csv)
        self.label_num = len(self.index_dict)
        
        # Calculate class frequencies if balanced sampling is enabled
        # if self.balanced_sampling:
        #     self.class_counts = torch.zeros(self.label_num)
        #     for datum in self.data:
        #         for label_str in datum['labels'].split(','):
        #             self.class_counts[int(self.index_dict[label_str])] += 1
        #     self.class_weights = 1.0 / (self.class_counts + 1e-7)
        
        # Audio config parameters
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.freqm = self.audio_conf.get('freqm')
        self.timem = self.audio_conf.get('timem')
        self.mixup = self.audio_conf.get('mixup')
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        self.skip_norm = self.audio_conf.get('skip_norm', False)
        self.noise = self.audio_conf.get('noise', False)
        self.freq_division_mode = self.audio_conf.get('freq_division_mode', 'uniform')
        self.split_freq = self.audio_conf.get('split_freq', 1000)

    def _make_index_dict(self, label_csv):
        index_lookup = {}
        with open(label_csv, 'r') as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                index_lookup[row['mid']] = row['index']
        return index_lookup

    def _wav2fbank(self, filename, filename2=None):
        # Get whether to resample from config
        resample = self.audio_conf.get('resample', False)
        target_sample_rate = self.audio_conf.get('sample_rate', 16000)

        # Load waveform and resample (if enabled)
        waveform, sr = resample_audio(filename, target_sample_rate=target_sample_rate, resample=resample)

        # Get whether to filter from config
        apply_filter = self.audio_conf.get('filter', False)
        cutoff = self.audio_conf.get('cutoff_freq', 1024)
        if apply_filter:
            waveform_np = waveform.numpy()
            waveform_np = low_pass_filter(waveform_np, sr, cutoff).copy()
            waveform = torch.from_numpy(waveform_np)

        waveform = waveform - waveform.mean()

        # If a second file is provided, perform mixup
        if filename2:
            waveform2, sr2 = torchaudio.load(filename2)
            if resample:
                resampler = torchaudio.transforms.Resample(orig_freq=sr2, new_freq=target_sample_rate)
                waveform2 = resampler(waveform2)

            waveform2 = waveform2 - waveform2.mean()
            mix_lambda = np.random.beta(10, 10)
            waveform = mix_lambda * waveform + (1 - mix_lambda) * waveform2
            waveform = waveform - waveform.mean()

        # Extract fbank features based on frequency division mode
        if self.freq_division_mode == 'uniform':
            # Use standard kaldi fbank with uniform mel-scale bins
            fbank = torchaudio.compliance.kaldi.fbank(
                waveform,
                htk_compat=True,
                sample_frequency=sr,
                use_energy=False,
                window_type='hanning',
                num_mel_bins=self.melbins,
                dither=0.0,
                frame_shift=10
            )
        elif self.freq_division_mode == 'split_1khz':
            # Use custom mel filterbank with split at 1kHz
            n_fft = 512
            win_length = int(sr * 0.025)  # 25ms window
            hop_length = int(sr * 0.010)  # 10ms hop (frame_shift=10ms)
            
            # Compute STFT
            stft = torch.stft(
                waveform[0],
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=torch.hann_window(win_length),
                center=True,
                pad_mode='reflect',
                normalized=False,
                onesided=True,
                return_complex=True
            )
            
            # Compute power spectrogram
            power_spec = torch.abs(stft) ** 2
            
            # Get custom mel filterbank
            mel_fb = get_custom_mel_filterbank(
                sample_rate=sr,
                n_fft=n_fft,
                n_mels=self.melbins,
                mode='split_1khz',
                split_freq=self.split_freq
            )
            
            # Apply mel filterbank
            mel_spec = torch.matmul(mel_fb, power_spec)
            
            # Convert to log scale (to match kaldi fbank output)
            fbank = torch.log(mel_spec + 1e-6).transpose(0, 1)
        else:
            raise ValueError(f"Unknown freq_division_mode: {self.freq_division_mode}")

        # Pad or trim fbank
        target_length = self.audio_conf.get('target_length')
        n_frames = fbank.shape[0]
        p = target_length - n_frames
        if p > 0:
            fbank = torch.nn.functional.pad(fbank, (0, 0, 0, p), mode='constant')
        elif p < 0:
            fbank = fbank[:target_length, :]

        return fbank, 0 if filename2 is None else mix_lambda
    
    def __getitem__(self, index):
        # Original __getitem__ implementation
        if random.random() < self.mixup:
            datum = self.data[index]
            mix_sample_idx = random.randint(0, len(self.data)-1)
            mix_datum = self.data[mix_sample_idx]
            fbank, mix_lambda = self._wav2fbank(datum['wav'], mix_datum['wav'])
            
            label_indices = np.zeros(self.label_num)
            for label_str in datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] += mix_lambda
            for label_str in mix_datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] += 1.0-mix_lambda
            label_indices = torch.FloatTensor(label_indices)
            
            # Calculate weight for mixed sample
            # if self.balanced_sampling:
            #     weight = 0.0
            #     for label_str in datum['labels'].split(','):
            #         weight += self.class_weights[int(self.index_dict[label_str])] * mix_lambda
            #     for label_str in mix_datum['labels'].split(','):
            #         weight += self.class_weights[int(self.index_dict[label_str])] * (1.0 - mix_lambda)
            #     weight = torch.tensor(weight)
        else:
            datum = self.data[index]
            label_indices = np.zeros(self.label_num)
            fbank, mix_lambda = self._wav2fbank(datum['wav'])
            for label_str in datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] = 1.0
            label_indices = torch.FloatTensor(label_indices)
            
            # Calculate weight for single sample
            # if self.balanced_sampling:
            #     weight = 0.0
            #     for label_str in datum['labels'].split(','):
            #         weight += self.class_weights[int(self.index_dict[label_str])]
            #     weight = torch.tensor(weight)
        
        # Apply SpecAug
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        fbank = torch.transpose(fbank, 0, 1).unsqueeze(0)
        
        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank)
        
        fbank = fbank.squeeze(0).transpose(0, 1)
        
        if not self.skip_norm:
            fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        
        if self.noise:
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)
        
        # if self.balanced_sampling:
        #     return fbank, label_indices, weight
        # else:
        return fbank, label_indices

    def __len__(self):
        return len(self.data)
