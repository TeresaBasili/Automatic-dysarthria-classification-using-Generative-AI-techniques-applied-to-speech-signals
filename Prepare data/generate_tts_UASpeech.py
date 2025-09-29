### Modified version of generate_tts_preds.py for generating speech with texts from UA-Speech dataset.
### generate_tts_preds.py for generating speech with texts from TORGO dataset (availeble at https://github.com/WingZLeung/TTDS?tab=readme-ov-file)

import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from omegaconf import DictConfig, OmegaConf
import hydra

from model import GradTTS
from utils import save_plot
from text.symbols import symbols

# from nemo.collections.tts.models import HifiGanModel
#from speechbrain.pretrained import HIFIGAN
try:
    from speechbrain.inference import HIFIGAN
    print("Successfully imported HIFIGAN from speechbrain.pretrained")
except ImportError as e:
    print(f"ImportError: {e}")
    print("Trying to import HIFIGAN from speechbrain.inference.vocoders")
    try:
        from speechbrain.inference.vocoders import HIFIGAN
        print("Successfully imported HIFIGAN from speechbrain.inference.vocoders")
    except ImportError as e:
        print(f"ImportError: {e}")
        print("Failed to import HIFIGAN from both sources. Please check your installations.")
        raise
        
from scipy.io.wavfile import write

from utils import intersperse, save_plot
from text import text_to_sequence, cmudict
import soundfile as sf

@hydra.main(version_base=None, config_path='./config')
def main(cfg: DictConfig):
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    device = torch.device(f'cuda:{cfg.training.gpu}')


    print('Initializing model...')
    model = GradTTS(cfg)
    model.load_state_dict(torch.load(cfg.eval.checkpoint, map_location=lambda loc, storage: loc))
    model.to(device).eval()
    print('Number of encoder parameters = %.2fm' % (model.encoder.nparams/1e6))
    print('Number of decoder parameters = %.2fm' % (model.decoder.nparams/1e6))

    print('Initializing vocoder...')
    # vocoder = HifiGanModel.from_pretrained(model_name='nvidia/tts_hifigan')
    vocoder = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-16kHz")

    print(f'Synthesizing text...', end=' ')

    prompt = pd.read_excel('lista_parole.xlsx', header=None) # .xlsx with UA-Speech text (765 words)

    cmu = cmudict.CMUDict(cfg.data.cmudict_path)

    for _, row in prompt.iterrows():

        text=row[0]
        cod=row[1]
    
        x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))).to(device)[None]
        x_lengths = torch.LongTensor([x.shape[-1]]).to(device)
        spk = cfg.eval.spk
        spk = torch.tensor(spk).to(device)

        y_enc, y_dec, attn = model.forward(x, x_lengths, n_timesteps=cfg.eval.timesteps, spk=spk)

        # audio = vocoder.convert_spectrogram_to_audio(spec=y_dec)
        audio = vocoder.decode_batch(y_dec)
        audio = audio.squeeze().to('cpu').detach().numpy()

        out = f'{cfg.eval.out_dir}' # path output directory
        os.makedirs(out, exist_ok=True)

        out_path = f'{cfg.eval.out_dir}/{cod}.wav'

        sf.write(out_path, audio, 16000)


if __name__ == '__main__':
    main()