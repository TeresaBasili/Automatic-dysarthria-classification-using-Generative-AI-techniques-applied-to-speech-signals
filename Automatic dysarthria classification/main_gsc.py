import os, sys
import numpy as np
from easydict import EasyDict
from dataset import gsc_melspectrogram
from datetime import datetime
import argparse
import pandas as pd
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchsummary import summary 
torch.manual_seed(184)


class Adversarial_Reprogramming(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.init_dataset()
        ############################################################################################################
        
        #mobile net
        #self.model = torchvision.models.mobilenet_v3_small().to(cfg.device)
        #self.num_features = self.model.classifier[-1].in_features
        #self.model.classifier[-1] = torch.nn.Linear(self.num_features, 4).to(cfg.device)

        #resnet152
        #self.model = torchvision.models.resnet152().to(cfg.device)
        #self.in_features = self.model.fc.in_features
        #self.model.fc = nn.Linear(self.in_features, 4).to(cfg.device)
        
        #resnet50
        self.model = torchvision.models.resnet50().to(cfg.device)
        self.num_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(self.num_features, 4).to(cfg.device)
        
        ############################################################################################################
        
        self.criterion = torch.nn.CrossEntropyLoss().to(cfg.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr) 
        self.writer = SummaryWriter(log_dir=os.path.join(cfg.log_dir, cfg.experiment_id))
        self.opt_scheduler = ExponentialLR(self.optimizer, gamma=0.96**(1/5))
        

    def init_dataset(self):
        df = pd.read_csv(self.cfg.meta_file)
        #train_df = df[df['split'] == 'train']
        train_df=df[df['split'].isin(['train', 'val'])]
        print(f'Loaded {len(train_df)} items for training')
        validation_df = df[df['split'] == 'test']
        print(f'Loaded {len(validation_df)} items for validation')
        #test_df = df[df['split'] == 'test']
        #print(f'Loaded {len(test_df)} items for test')
        train_set = gsc_melspectrogram(train_df, self.cfg)
        val_set = gsc_melspectrogram(validation_df, self.cfg)
        #test_set = gsc_melspectrogram(test_df, self.cfg)

        self.train_loader = torch.utils.data.DataLoader(train_set,
                                                        batch_size=self.cfg.batch_size,
                                                        shuffle=True,
                                                        num_workers=0,
                                                        drop_last=True)
        self.val_loader = torch.utils.data.DataLoader(val_set,
                                                        batch_size=1,
                                                        shuffle=False,
                                                        num_workers=0,
                                                        drop_last=True)
        # self.test_loader = torch.utils.data.DataLoader(test_set,
        #                                                 batch_size=1,
        #                                                 shuffle=False,
        #                                                 num_workers=0,
        #                                                 drop_last=True)
    
    


    def validate(self):
        print('validate...')
        y_true=[]
        y_pred=[]
        cum_acc = 0.0
        patient_acc = [0]*15 
        patient_count = [0]*15
        cum_loss_val = 0.0
        self.model.eval()
        with torch.no_grad():
            for image, label, patient_id in tqdm(self.val_loader):
                image = image.to(self.cfg.device)
                label = label.to(self.cfg.device)
                image = image.repeat(1, 3, 1, 1)

                out = self.model(image[:,:,:self.cfg.h2, :])
                loss_val = self.criterion(out, label)
                cum_loss_val += loss_val.item()
                _, pred = torch.max(out, 1)

                y_true.append(label.to('cpu'))
                y_pred.append(pred.to('cpu'))
                correct = (pred == label).sum().item()
                patient_acc[patient_id] += correct
                patient_count[patient_id] += 1
                cum_acc += correct
            macro_f1 = f1_score(y_true, y_pred, average='macro')
            micro_f1 = f1_score(y_true, y_pred, average='micro')
            avg_f1 = f1_score(y_true, y_pred, average='weighted')
            acc = accuracy_score(y_true, y_pred)
            result_test= {
                'Accuracy': acc,
                'macro F1-score' : macro_f1,
                'micro F1-score' : micro_f1,
                'weighted F1-score' : avg_f1,
            }
        return cum_acc, cum_loss_val, patient_acc, patient_count, result_test


    def train(self):
        self.best_accuracy = 0.0
        self.best_acc_patient = 0.0

        for e in range(1, self.cfg.epochs+1):
            self.model.train()
            cum_loss = 0.0
            print(f'Epoch {e}/{self.cfg.epochs}')
            for it, (image, label, _) in enumerate(tqdm(self.train_loader)):
                image = image.to(self.cfg.device)
                label = label.to(self.cfg.device)
                image = image.repeat(1, 3, 1, 1)

                out = self.model(image[:,:,:self.cfg.h2, :])
                loss = self.criterion(out, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                cum_loss += loss.item()
            self.opt_scheduler.step()
            cum_loss /= len(self.train_loader)

            # Validation
            val_accuracy, cum_loss_val, patient_acc, patient_count, result_test  = self.validate()
            cum_loss_val /= len(self.val_loader)   
            val_accuracy /= len(self.val_loader)
            acc_patient = [acc/count for acc, count in zip(patient_acc, patient_count) if count > 0]
            print(f'Patient accuracy: {acc_patient}')
            print(f'Train loss {cum_loss:.2f} - Validation accuracy: {100*(val_accuracy):.2f} % - Validation loss {cum_loss_val:.2f}')
           
            # log to tensorboard
            self.writer.add_scalar('TRAIN_LOSS', cum_loss, e)
            self.writer.add_scalar('VAL_ACCURACY', val_accuracy, e)
            self.writer.add_scalar('VAL_LOSS', cum_loss_val, e)
            self.writer.add_scalar('LEARNING_RATE', self.optimizer.param_groups[0]["lr"], e)
            #self.writer.add_scalars('PATIENT_ACCURACY', {f'patient_{i}': acc for i, acc in enumerate(acc_patient)}, e)
            self.writer.flush()

            # save the model checkpoint with highest accuracy
            if val_accuracy > self.best_accuracy:
                torch.save(self.model.state_dict(), os.path.join(self.cfg.log_dir, self.cfg.experiment_id, 'best_ckeckpoint.pth'))
                torch.save(acc_patient, os.path.join(self.cfg.log_dir, self.cfg.experiment_id, 'acc_patient_best.pth'))
                torch.save(val_accuracy, os.path.join(self.cfg.log_dir, self.cfg.experiment_id, 'acc_best.pth'))
                print('>>best checkpoint saved')
                self.best_accuracy = val_accuracy
                self.best_epoch = e
                path_result=os.path.join(self.cfg.log_dir, self.cfg.experiment_id, 'test_result.json')
                with open(path_result, 'w') as file:
                    json.dump(result_test, file)
            print('')

        self.writer.close()

    def test(self):
        print('test...')
        y_true=[]
        y_pred=[]
        cum_acc = 0.0
        patient_acc = [0]*15 
        patient_count = [0]*15 
        cum_loss_test = 0.0
        self.model.eval()
        with torch.no_grad():
            for image, label, patient_id in tqdm(self.test_loader):
                image = image.to(self.cfg.device)
                label = label.to(self.cfg.device)
                image = image.repeat(1, 3, 1, 1)
                out = self.model(image)
                loss_test = self.criterion(out, label)
                cum_loss_test += loss_test.item()
                _, pred = torch.max(out, 1)
                y_true.append(label.to('cpu'))
                y_pred.append(pred.to('cpu'))
                correct = (pred == label).sum().item()
                patient_acc[patient_id] += correct
                patient_count[patient_id] += 1
                cum_acc += correct
            test_loss = cum_loss_test / len(self.test_loader) 
            combined_true = torch.cat(y_true)
            combined_pred = torch.cat(y_pred)
            np_true=combined_true.numpy()
            np_pred=combined_pred.numpy()
            ConfusionMatrixDisplay.from_predictions(y_true=np_true, y_pred=np_pred, labels=[0, 1, 2, 3])
            plt.savefig(os.path.join(self.cfg.log_dir, self.cfg.experiment_id, 'ConfusionMatrix.pdf'))
            macro_f1 = f1_score(y_true, y_pred, average='macro')
            micro_f1 = f1_score(y_true, y_pred, average='micro')
            avg_f1 = f1_score(y_true, y_pred, average='weighted')
            acc = accuracy_score(y_true, y_pred)
            print(f"Test Loss: {test_loss:.3f} | Test Macro F1: {macro_f1:.3f} | Test Micro F1: {micro_f1:.3f} | Test Weighted F1: {avg_f1:.3f} | Test Acc: {100*acc:.2f}%")
            result_test= {
                'Accuracy': acc,
                'Loss': test_loss,
                'macro F1-score' : macro_f1,
                'micro F1-score' : micro_f1,
                'weighted F1-score' : avg_f1,
            }
            path_result=os.path.join(self.cfg.log_dir, self.cfg.experiment_id, 'test_result.json')
            with open(path_result, 'w') as file:
                json.dump(result_test, file)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='gsc', type=str, choices=['gsc'])
    cli_args = parser.parse_args()
    if cli_args.cfg == 'gsc':
       from config_gsc import cfg
    else:
        raise NotImplementedError()
       
    
    AR = Adversarial_Reprogramming(cfg)

    print(f'Experiment {cfg.experiment_id}, start Training...')
    start_tstamp = datetime.now()
    AR.train()
    end_tstamp = datetime.now()
    print(f'Experiment {cfg.experiment_id}, Training started at {start_tstamp.strftime("%m/%d/%Y - %H:%M:%S")}, Training finished at {end_tstamp.strftime("%m/%d/%Y - %H:%M:%S")}')
    print(f'Best accuracy: {100*(AR.best_accuracy):.2f} % at epoch {AR.best_epoch}')
    #AR.test()
    
    