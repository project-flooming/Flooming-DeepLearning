import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from torchcallback import CheckPoint, EarlyStopping
from model.classification_model import VGG19

device = torch.device('cuda')

model = VGG19().to(device)

es_save_path = './model/es_checkpoint.pt'
cp_save_path = './model/cp_checkpoint.pt'
checkpoint = CheckPoint(verbose=True, path=cp_save_path)
early_stopping = EarlyStopping(patience=20, verbose=True, path=es_save_path)

lr = 0.0001
beta1 = 0.9
beta2 = 0.999
loss_func = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1,beta2))

def valid_step(model,
               validation_data):
    model.eval()
    with torch.no_grad():
        vbatch_loss, vbatch_acc = 0, 0
        for vbatch, (val_images, val_labels) in enumerate(validation_data):
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            
            val_outputs = model(val_images)
            val_loss = loss_func(val_outputs, val_labels)
            output_index = torch.argmax(val_outputs, dim=1)
            val_acc = (output_index==val_labels).sum()/len(val_outputs)
            
            vbatch_loss += val_loss.item()
            vbatch_acc += val_acc.item()
            
            del val_images; del val_labels; del val_outputs
            torch.cuda.empty_cache()
            
    return vbatch_loss/(vbatch+1), vbatch_acc/(vbatch+1)

def train_on_batch(model, train_data):
    batch_loss, batch_acc = 0, 0
    for batch, (train_images, train_labels) in enumerate(train_data):
        model.train()

        train_images = train_images.to(device)
        train_labels = train_labels.to(device)

        optimizer.zero_grad()

        train_outputs = model(train_images)
        loss = loss_func(train_outputs, train_labels)
        output_index = torch.argmax(train_outputs, dim=1)
        acc = (output_index==train_labels).sum()/len(train_outputs)

        batch_loss += loss.item()
        batch_acc += acc.item()

        loss.backward()
        optimizer.step()

        del train_images; del train_labels; del train_outputs
        torch.cuda.empty_cache()

    return batch_loss/(batch+1), batch_acc/(batch+1)

def train_step(model,
               train_data,
               validation_data,
               epochs,
               learning_rate_scheduler=False,
               check_point=False,
               early_stop=False,
               last_epoch_save_path='./model/last_checkpoint.pt'):
    
    loss_list, acc_list = [], []
    val_loss_list, val_acc_list = [], []
    
    print('Start Model Training...!')
    start_training = time.time()
    for epoch in tqdm(range(epochs)):
        init_time = time.time()
        batch_loss, batch_acc = 0, 0
        
        train_loss, train_acc = train_on_batch(model, train_data)
        loss_list.append(train_loss)
        acc_list.append(train_acc)
            
        val_loss, val_acc = valid_step(model, validation_data)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        
        end_time = time.time()
        
        print(f'\n[Epoch {epoch+1}/{epochs}]'
              f'  [time: {end_time-init_time:.3f}s]'
              f'  [lr = {optimizer.param_groups[0]["lr"]}]')
        print(f'[train loss: {train_loss:.3f}]'
              f'  [train acc: {train_acc:.3f}]'
              f'  [valid loss: {val_loss:.3f}]'
              f'  [valid acc: {val_acc:.3f}]')
            
        if check_point:
            checkpoint(val_loss, model)
            
        if early_stop:
            assert check_point==False, \
                'Choose between Early Stopping and Check Point'
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print('\n##########################\n'
                      '##### Early Stopping #####\n'
                      '##########################')
                break
                
    if early_stop==False and check_point==False:
        torch.save(model.state_dict(), last_epoch_save_path)
        print('Saving model of last epoch.')
        
    end_training = time.time()
    print(f'\nTotal time for training is {end_training-start_training:.3f}s')
    
    return {
        'model': model, 
        'loss': loss_list, 
        'acc': acc_list, 
        'val_loss': val_loss_list, 
        'val_acc': val_acc_list
    }