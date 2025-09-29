# Created on: 12/24/2024
# Author: Carlo Aironi
# Google Speech Command dataset class

import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from torch.utils.data import Dataset
import librosa
import numpy as np
import os
import noisereduce as nr
import matplotlib.pyplot as plt 
class gsc_melspectrogram(Dataset):
    """
    """

    def __init__(self, meta_df, cfg):
        self.meta_df = meta_df
        self.cfg = cfg

        self.mel_spectrogram = T.MelSpectrogram(sample_rate=16000,
                                                n_fft=1024,
                                                hop_length=cfg.hop_length,
                                                power=1.0,
                                                normalized=True,
                                                n_mels=90).to(self.cfg.device)

    
    def pad_audio(self, file_path, duration, sr=16000):
        y, sr = librosa.load((file_path), sr=sr)
        #y = nr.reduce_noise(y, sr, device= self.cfg.device)
        # Calcola il numero di campioni necessari
        n_samples = int(duration * sr)
        # Calcola il numero di campioni attuali
        current_samples = len(y) #TODO: rendi i campioni sempre pari
        # Se il numero di campioni attuali è minore di quelli necessari
        if current_samples < n_samples:
            # Calcola il numero di campioni da aggiungere
            n_samples_to_add = n_samples - current_samples
            # Calcola il numero di campioni da aggiungere a sinistra e a destra
            n_samples_to_add_left = n_samples_to_add // 2
            n_samples_to_add_right = n_samples_to_add - n_samples_to_add_left 
            # Genera un array di zeri
            padding_left = np.zeros(n_samples_to_add_left, dtype='float32')
            padding_right = np.zeros(n_samples_to_add_right,  dtype='float32')
            # Concatena il padding al segnale audio
            y_padded = np.concatenate([padding_left, y, padding_right])
        # Se il numero di campioni attuali è maggiore di quelli necessari
        elif current_samples > n_samples:
            # Calcola il numero di campioni da rimuovere
            n_samples_to_remove = current_samples - n_samples
            # Calcola il numero di campioni da rimuovere a sinistra e a destra
            n_samples_to_remove_left = n_samples_to_remove // 2
            n_samples_to_remove_right = n_samples_to_remove - n_samples_to_remove_left
            # Trimma il segnale audio
            y_padded = y[n_samples_to_remove_left:-n_samples_to_remove_right]
        # Se il numero di campioni attuali è uguale a quelli necessari
        else:
            y_padded = y
        # Plot del segnale audio paddato/trimmato
        
        
        return torch.from_numpy(y_padded).unsqueeze(0)



    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, idx):
        # load the waveform
        waveform = self.pad_audio(self.meta_df.iloc[idx]['filename'], 2.86)
        waveform = waveform.to(self.cfg.device)
        lbl = torch.tensor(self.meta_df.iloc[idx]['label'], dtype=torch.long)
        lbl = lbl.to(self.cfg.device)
        id = torch.tensor(self.meta_df.iloc[idx]['id'], dtype=torch.int64)
        id = id.to(self.cfg.device)
        # compute the spectrogram
        S = self.mel_spectrogram(waveform)
        
        if self.cfg.scale == 'logmel':
            S = torch.log10(S + 1e-9)

        # normalization in [0,1]
        S_min = S.min()
        S_max = S.max()
        S_norm = (S - S_min) / (S_max - S_min)
        
        return (S_norm, lbl, id)
    
if __name__ == '__main__':
    import pandas as pd 
    from config_gsc import cfg
    import matplotlib.pyplot as plt

    df = pd.read_csv(cfg.meta_file)
    # Create dataframes for each split
    train_df = df[df['split'] == 'train']
    print(f'Loaded {len(train_df)} items for training')
    validation_df = df[df['split'] == 'validation']
    print(f'Loaded {len(validation_df)} items for validation')
    test_df = df[df['split'] == 'test']
    print(f'Loaded {len(test_df)} items for test')
    train_set = gsc_melspectrogram(train_df, cfg)
    test_set = gsc_melspectrogram(test_df, cfg)
    #for d,l,_ in train_set:
    #    print(d.shape, l)

    ### PLOT ###

    for i in range(2798, 2810):
        wave_path=train_df['filename'].iloc[i]
        #print(wave_path)
        waveform = train_set.pad_audio(file_path=wave_path, duration=2.86)
        waveform = waveform.squeeze(0)
        plt.figure(figsize=(10, 3))
        plt.plot(waveform.numpy(), color='blue')
        plt.title('Waveform del segnale audio')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"waveform_image_{i}.png")
        plt.close()

        image, label, spk = train_set[i]
        image_np = image.detach().cpu().numpy()
        image_2d = image_np.squeeze(0)
        print(label)
        plt.figure(figsize=(6, 4))
        plt.imshow(image_2d, aspect='auto', origin='lower', cmap='magma') 
        plt.colorbar(label='Amplitude')       
        plt.title(f"Spettrogramma Mel")
        plt.xlabel('Time')
        plt.ylabel('Mel bins')
        plt.tight_layout()
        plt.savefig(f"mel_image_{i}.png")
        plt.close()







        
   