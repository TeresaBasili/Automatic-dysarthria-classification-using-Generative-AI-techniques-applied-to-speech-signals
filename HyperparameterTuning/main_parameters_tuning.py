import json
import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset import gsc_melspectrogram
from random_parameter import get_parameter
import gc
import numpy as np
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)



def main(lr_adam, lr_sgd, mel_bands, opt, hop_length, model, batch_size, epochs, patience, binary=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    writer = SummaryWriter(log_dir=os.path.join(model_path, str(i)))
    
    #lr_value, mel_value, opt_value, hl_value = get_parameter(lr_adam, lr_sgd, mel_bands, opt, hop_length)
    lr_value=lr_adam
    mel_value=mel_bands
    opt_value=opt
    hl_value=hop_length
    print(f"Using model: {model}, Learning Rate: {lr_value}, Mel Bands: {mel_value}, Optimizer: {opt_value}, Hop Length: {hl_value}")


    #Inizializzo il modello
    if binary:
        if model == 'mobilenet':
            model = torchvision.models.mobilenet_v3_small().to(device)
            num_features = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(num_features, 1).to(device)
            #csv_path = '/home/tbasili/TTDS/UA-Speech/file_csv/UAS+TORGO_binary.csv' 
            #seconds_length = 1.90
            csv_path = '/home/tbasili/TTDS/UA-Speech/file_csv/UAS_binary.csv'
            seconds_length = 2.96
        elif model == 'resnet50':
            model = torchvision.models.resnet50().to(device)
            num_features = model.fc.in_features
            model.fc = torch.nn.Linear(num_features, 1).to(device)
            #csv_path = '/home/tbasili/TTDS/UA-Speech/file_csv/UAS+syn_binary.csv' 
            #seconds_length = 2.31
            csv_path = '/home/tbasili/TTDS/UA-Speech/file_csv/UAS_binary.csv'
            seconds_length = 2.96
            #csv_path = '/home/tbasili/TTDS/UA-Speech/file_csv/UAS+TORGO+allsyn_binary.csv' 
            #seconds_length = 1.42
            #csv_path = '/home/tbasili/TTDS/UA-Speech/file_csv/UAS+TORGO_binary.csv'
            #seconds_length = 1.90

        elif model == 'resnet152':
            model = torchvision.models.resnet152().to(device)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, 1).to(device)
            #csv_path = '/home/tbasili/TTDS/UA-Speech/file_csv/UAS+TORGO_binary.csv' 
            #seconds_length = 1.90
            csv_path = '/home/tbasili/TTDS/UA-Speech/file_csv/UAS_binary.csv'
            seconds_length = 2.96
            #csv_path = '/home/tbasili/TTDS/UA-Speech/file_csv/UAS+TORGO+allsyn_binary.csv' 
            #seconds_length = 1.42
            #csv_path = '/home/tbasili/TTDS/UA-Speech/file_csv/UAS+syn_binary.csv' 
            #seconds_length = 2.31
            
    else:
        if model == 'mobilenet':
            model = torchvision.models.mobilenet_v3_small().to(device)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, 4).to(device)
            #csv_path = '/home/tbasili/TTDS/UA-Speech/file_csv/UAS+TORGO_multiclass.csv'
            csv_path = '/home/tbasili/TTDS/UA-Speech/file_csv/UAS_train_val_test_filtered.csv'  
            #seconds_length = 1.90
            seconds_length = 2.86
        elif model == 'resnet50':
            model = torchvision.models.resnet50().to(device)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, 4).to(device)
            #csv_path = '/home/tbasili/TTDS/UA-Speech/file_csv/UAS+syn_split_filtered.csv'
            #seconds_length = 2.31
            csv_path = '/home/tbasili/TTDS/UA-Speech/file_csv/UAS_train_val_test_filtered.csv' 
            seconds_length = 2.86
            #csv_path = '/home/tbasili/TTDS/UA-Speech/file_csv/UAS+TORGO+allsyn_multiclass.csv' 
            #seconds_length = 1.42
            #csv_path = '/home/tbasili/TTDS/UA-Speech/file_csv/UAS+TORGO_multiclass.csv'
            #seconds_length = 1.90
        elif model == 'resnet152':
            model = torchvision.models.resnet152().to(device)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, 4).to(device)
            #csv_path = '/home/tbasili/TTDS/UA-Speech/file_csv/UAS+TORGO_multiclass.csv' 
            #seconds_length = 1.90
            csv_path = '/home/tbasili/TTDS/UA-Speech/file_csv/UAS_train_val_test_filtered.csv' 
            seconds_length = 2.86
            #csv_path = '/home/tbasili/TTDS/UA-Speech/file_csv/UAS+TORGO+allsyn_multiclass.csv' 
            #seconds_length = 1.42
            #csv_path = '/home/tbasili/TTDS/UA-Speech/file_csv/UAS+syn_split_filtered.csv'
            #seconds_length = 2.31

    #definizione del dataset

    df = pd.read_csv(csv_path)
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']


    train_set = gsc_melspectrogram(train_df, hop_length=hl_value, seconds_lenght=seconds_length)
    val_set = gsc_melspectrogram(val_df, hop_length=hl_value, seconds_lenght=seconds_length)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, shuffle=False, drop_last=True)

    if binary:
        criterion = torch.nn.BCEWithLogitsLoss().to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    if opt_value == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_value)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr_value, momentum=0.9)

    opt_scheduler = ExponentialLR(optimizer, gamma = 0.96**(1/5))



    #training loop
    best_f1 = 0.0
    best_e = 0
    for e in range(1, epochs+1):
        model.train()
        cum_loss = 0.0
        for image, label, _ in tqdm(train_loader, desc=f"Epoch {e}/{epochs}"):
            image, label = image.to(device), label.to(device)
            if binary:
                label=label.float()
                label = label.unsqueeze(1)
            image = image.repeat(1, 3, 1, 1)
            out = model(image[:,:,:mel_value, :])
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cum_loss += loss.item()
        opt_scheduler.step()
        cum_loss /= len(train_loader)
        # Validation

        print("Validating...")


        y_true, y_pred = [], []
        cum_loss_val = 0.0
        model.eval()
        with torch.no_grad():
            for image, label, _ in tqdm(val_loader, desc="Validating"):
                image, label = image.to(device), label.to(device)
                image = image.repeat(1, 3, 1, 1)
                if binary:
                    label=label.float()
                    label = label.unsqueeze(1)
                out = model(image[:,:,:mel_value, :])
                loss_val = criterion(out, label)
                cum_loss_val += loss_val.item()
                if binary:
                    prob  = torch.sigmoid(out)                # (batch_size,1)
                    pred  = (prob > 0.5).long()
                else:
                    _, pred = torch.max(out, 1)
                y_true.append(label.cpu())
                y_pred.append(pred.cpu())
        val_loss = cum_loss_val / len(val_loader)
        y_true = torch.cat(y_true).numpy()
        y_pred = torch.cat(y_pred).numpy()
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        avg_f1 = f1_score(y_true, y_pred, average='weighted')
        acc = accuracy_score(y_true, y_pred)

        print(f"[Epoch {e}] Train Loss: {cum_loss:.3f} | Val Loss: {val_loss:.3f} | Val Macro F1: {macro_f1:.3f} | Val Micro F1: {micro_f1:.3f} | Val Weighted F1: {avg_f1:.3f} | Val Acc: {100*acc:.2f}%")

        writer.add_scalar('TRAIN_LOSS', cum_loss, e)
        writer.add_scalar('VAL_LOSS', val_loss, e)
        writer.add_scalar('VAL_MACRO_F1', macro_f1, e)
        writer.add_scalar('VAL_MICRO_F1', micro_f1, e)
        writer.add_scalar('VAL_WEIGHTED_F1', avg_f1, e)
        writer.add_scalar('VAL_ACC', acc, e)
        writer.add_scalar('LEARNING_RATE', optimizer.param_groups[0]["lr"], e)

        # Early stopping sul macro F1
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            patience_counter = 0
            best_e = e
            torch.save(model.state_dict(), os.path.join(model_path, str(i), 'best_model.pth'))
            print(">> New best model saved based on Macro F1 at epoch:", e )
        else:
            if e > 15:
                patience_counter += 1
                print(f">> No improvement. Patience {patience_counter}/{patience}")
            else:
                patience_counter = 0
        if patience_counter >= patience:
            print(">> Early stopping triggered.")
            break
    writer.close()

    parameters = {
    'learning_rate': lr_value,
    'mel_bands': mel_value,
    'optimizer': opt_value,
    'hop_length': hl_value,
    'best_f1':best_f1,
    'best_e' : best_e}
    
    with open(percorso_file, 'w') as file:
        json.dump(parameters, file)



if __name__ == "__main__":

    waights_path = ''
   


    iter = 18
    batch_size = 64
    epochs = 100
    patience = 15
    # definizione degli iperparametri
 
    
    lr_adam = [ 0.01,0.05]  
    #lr_sgd = [0.001, 0.005]
    mel_bands = [30, 60, 90] 
    opt = ['Adam']  
    hop_length = [64, 128, 256] 
    models = ['resnet50', 'resnet152'], 
    

    for model in models:
        i=0

        if not os.path.exists(os.path.join(waights_path, str(model))):
            os.makedirs(os.path.join(waights_path, str(model)))
        model_path=os.path.join(waights_path, str(model))
        

        # for i in range(iter):
        #     percorso_file = os.path.join(model_path, str(i),'parameters.json')

        #     if not os.path.exists(os.path.join(model_path, str(i))):
        #         os.makedirs(os.path.join(model_path, str(i)))

        #    
        #     print(f'iteration: {i}')
        #     main(lr_adam, lr_sgd, mel_bands, opt, hop_length, model, batch_size, epochs, patience, binary=True)
        
        for lr in lr_adam:
            for mel in mel_bands:
                for hl in hop_length:
                    if not os.path.exists(os.path.join(model_path, str(i))):
                        os.makedirs(os.path.join(model_path, str(i)))
        
                    percorso_file = os.path.join(model_path, str(i),'parameters.json')
                    main(lr, lr_sgd, mel, opt, hl, model, batch_size, epochs, patience, binary=True)
                    i+=1
                    torch.cuda.empty_cache()
                    gc.collect()
            


            # torch.cuda.empty_cache()
            # gc.collect()
    