import os
import torch
import torchaudio
import math
import random


class AudioDataset(torch.utils.data.Dataset):

    def __init__(self, clean_dir, chunk_size=16384, snr_range=(0,15), preemph_coef=0.95):
        super().__init__()
        self.clean_dir = clean_dir
        self.chunk_size = chunk_size
        self.snr_range = snr_range
        self.preemph_coef = preemph_coef

        self.file_paths = [
            os.path.join(self.clean_dir, file_name) for file_name in os.listdir(self.clean_dir) if file_name.endswith(".wav") 
        ]

        print(f"Dataset loaded with {len(self.file_paths)} files")

    def __len__(self):
        return len(self.file_paths)
    
    def _preemphasis(self, signal):
        padded_signal = torch.nn.functional.pad(signal, (1,0))
        preemphasized = padded_signal[:, 1:] - self.preemph_coef * padded_signal[:, :-1]
        return preemphasized
    
    def _add_stationnary_noise(self, clean_signal, snr_db):

        signal_power = torch.mean(clean_signal**2)
        if signal_power ==0:
            return clean_signal
        
        snr_linear = 10**(snr_db/10)
        noise_power = signal_power / snr_linear
        noise = torch.randn_like(clean_signal) * math.sqrt(noise_power)

        return clean_signal + noise
    
    def __getitem__(self, idx):
        
        file_path = self.file_paths[idx]
        waveform, sample_rate = torchaudio.load(file_path)

        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = waveform[0:1, :]

        signal_length = waveform.shape[1]
        if signal_length > self.chunk_size:
            max_start = signal_length - self.chunk_size
            start_idx = random.randint(0, max_start)
            extracted_chunk = waveform[:, start_idx:start_idx+self.chunk_size]
        else:
            extracted_chunk = torch.nn.functional.pad(waveform, (0, self.chunk_size - signal_length))

        clean_chunk = self._preemphasis(extracted_chunk)
        snr_db = random.uniform(self.snr_range[0], self.snr_range[1])
        noisy_chunk = self._add_stationnary_noise(clean_chunk, snr_db)

        return noisy_chunk, clean_chunk    


