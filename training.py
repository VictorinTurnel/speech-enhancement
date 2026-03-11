import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import AudioDataset
from model import DenoiserModel

def train(clean_audio_dir, checkpoints_dir, epochs=50, batch_size=16, lr =2e-4, l2 = True):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected device : {device}")

    dataset = AudioDataset(clean_audio_dir, chunk_size=16384, snr_range=(0, 15))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)
    print("Dataset Loaded")

    model = DenoiserModel().to(device)

    if l2:
        criterion = nn.MSELoss()
    else:
        criterion = nn.L1Loss()

    optimizer = optim.RMSprop(model.parameters(), lr=lr)

    samples_dir = os.path.join(checkpoints_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    for epoch in range(epochs):
        model.train()
        running_loss = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_inx, batch in enumerate(pbar):

            noisy_audio = batch[0].to(device)
            clean_audio = batch[1].to(device)

            optimizer.zero_grad()

            enhanced_audio = model(noisy_audio)
            loss = criterion(enhanced_audio, clean_audio)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

            if (epoch+1)%5 == 0 and batch_inx==0:
                sample_noisy = noisy_audio[0].detach().cpu()
                sample_clean = clean_audio[0].detach().cpu()
                sample_enhanced = enhanced_audio[0].detach().cpu()

                torchaudio.save(os.path.join(samples_dir, f"epoch_{epoch+1}_noisy.wav"), sample_noisy, 16000)
                torchaudio.save(os.path.join(samples_dir, f"epoch_{epoch+1}_clean.wav"), sample_clean, 16000)
                torchaudio.save(os.path.join(samples_dir, f"epoch_{epoch+1}_enhanced.wav"), sample_enhanced, 16000)
            

        epoch_loss = running_loss/len(dataloader)
        print(f"Epoch {epoch+1} | Average loss : {epoch_loss:.4}")

        os.makedirs(checkpoints_dir, exist_ok=True)
        save_path = os.path.join(checkpoints_dir, f"denoiser_model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)

    print("Training completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_audio_dir", type=str, default="voicebank/clean_trainset_wav", help="path to clean train set directory")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints", help="path to checkpoints")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
    parser.add_argument("--loss", type=str, default="l1", help="loss choice : l1 or l2")

    args = parser.parse_args()
    train(args.clean_audio_dir,
        args.checkpoints_dir, 
        args.epochs, 
        args.batch_size, 
        args.lr,
        True if args.loss == "l2" else False
    )
    
