# config parameters for UA_speech dataset
# Created on: 12/24/2024
# Author: Carlo Aironi

import os
import torch
from easydict import EasyDict

cfg = EasyDict()

cfg.experiment_id = ''              # name experiment 
cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
cfg.src_net = 'resnet50'            # model
cfg.meta_file = ''                  # filepath .csv train, validation and test dataset
cfg.scale = 'logmel'
cfg.log_dir = ''                    # path output directory 
cfg.batch_size = 64
cfg.h2 = 90                         # height input size for target dataset (Mel bands)
cfg.hop_length = 256                # hop length for mel spectrogram
cfg.mean = ()                       # GSC digits mean
cfg.std = ()                        # GSC digits std dev
cfg.lmd = 1e-6                      # lambda regularization factor
cfg.lr = 0.05                       # initial learning rate
cfg.epochs = 100




# create log folder
if not os.path.exists(os.path.join(cfg.log_dir, cfg.experiment_id)):
    os.makedirs(os.path.join(cfg.log_dir, cfg.experiment_id))
else:
    raise Exception(f'Log folder {os.path.join(cfg.log_dir, cfg.experiment_id)} already exists!')

torch.save(cfg, os.path.join(cfg.log_dir, cfg.experiment_id, 'config.pth'))