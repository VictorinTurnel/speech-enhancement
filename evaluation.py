import os
import torch
import torchaudio
import argparse
import numpy as np
from scipy.signal import lfilter
from pesq import pesq
from tqdm import tqdm

from model import DenoiserModel
from dataset import AudioDataset 

def deemphasis(signal, coef=0.95):

    return lfilter([1.0], [1.0, -coef], signal)

def calculate_snr(clean, noisy):
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean((clean - noisy) ** 2)
    if noise_power == 0:
        return 100.0
    return 10 * np.log10(signal_power / noise_power)

def evaluate(model_path, test_clean_dir, output_dir, chunk_size=16384):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Initializing evaluation on device: {device}")

    model = DenoiserModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print("[INFO] Model loaded successfully.")

    os.makedirs(output_dir, exist_ok=True)
    
    dataset = AudioDataset(test_clean_dir, chunk_size=chunk_size)

    pesq_scores_noisy = []
    pesq_scores_enhanced = []
    snr_scores_noisy = []
    snr_scores_enhanced = []

    file_paths = dataset.file_paths

    with torch.no_grad():
        for file_path in tqdm(file_paths, desc="Evaluating test files"):
            
            waveform, sr = torchaudio.load(file_path)
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
            waveform = waveform[0:1, :]
            
            signal_length = waveform.shape[1]
            
            pad_size = chunk_size - (signal_length % chunk_size)
            if pad_size != chunk_size:
                waveform = torch.nn.functional.pad(waveform, (0, pad_size))
            
            num_chunks = waveform.shape[1] // chunk_size
            chunks_clean = torch.split(waveform, chunk_size, dim=1)
            
            enhanced_chunks = []
            noisy_chunks_full = []
            
            for chunk in chunks_clean:
                clean_pre = dataset._preemphasis(chunk)
                noisy_pre = dataset._add_stationnary_noise(clean_pre, snr_db=5.0)
                
                noisy_chunks_full.append(noisy_pre)
                
                noisy_tensor = noisy_pre.unsqueeze(0).to(device)
                enhanced_tensor = model(noisy_tensor)
                
                enhanced_chunk = enhanced_tensor.squeeze(0).cpu()
                enhanced_chunks.append(enhanced_chunk)

            noisy_full = torch.cat(noisy_chunks_full, dim=1).numpy().squeeze()
            enhanced_full = torch.cat(enhanced_chunks, dim=1).numpy().squeeze()
            clean_full = waveform.numpy().squeeze()
            
            enhanced_full = deemphasis(enhanced_full, coef=0.95)
            noisy_full = deemphasis(noisy_full, coef=0.95)
            
            clean_full = clean_full[:signal_length]
            noisy_full = noisy_full[:signal_length]
            enhanced_full = enhanced_full[:signal_length]

            filename = os.path.basename(file_path)
            torchaudio.save(os.path.join(output_dir, f"noisy_{filename}"), torch.tensor(noisy_full).unsqueeze(0), 16000)
            torchaudio.save(os.path.join(output_dir, f"enhanced_{filename}"), torch.tensor(enhanced_full).unsqueeze(0), 16000)

            try:
                pesq_n = pesq(16000, clean_full, noisy_full, 'wb')
                pesq_e = pesq(16000, clean_full, enhanced_full, 'wb')
                
                pesq_scores_noisy.append(pesq_n)
                pesq_scores_enhanced.append(pesq_e)
            except:
                pass
            
            snr_scores_noisy.append(calculate_snr(clean_full, noisy_full))
            snr_scores_enhanced.append(calculate_snr(clean_full, enhanced_full))

    print("\n" + "="*50)
    print(" QUANTITATIVE EVALUATION RESULTS")
    print("="*50)
    print(f"PESQ (Noisy)    : {np.mean(pesq_scores_noisy):.3f}")
    print(f"PESQ (Enhanced) : {np.mean(pesq_scores_enhanced):.3f}")
    print("-" * 50)
    print(f"SNR  (Noisy)    : {np.mean(snr_scores_noisy):.3f} dB")
    print(f"SNR  (Enhanced) : {np.mean(snr_scores_enhanced):.3f} dB")
    print("="*50)
    print(f"[*] Audio files saved to: {output_dir}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SEGAN model performance.")
    parser.add_argument("--model_path", type=str, default="checkpoints/denoiser_model_epoch_50.pth", help="Path to the trained model checkpoint (.pth)")
    parser.add_argument("--test_audio_dir", type=str, default="voicebank/clean_testset_wav", help="Directory containing clean test audio files")
    parser.add_argument("--output_dir", type=str, default="results_audio", help="Directory to save the generated .wav files")
    
    args = parser.parse_args()
    evaluate(args.model_path, args.test_audio_dir, args.output_dir)