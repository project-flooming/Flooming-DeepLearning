import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from model import Generator, Discriminator

device = torch.device('cuda')

generator = Generator().to(device)
discriminator = Discriminator().to(device)

patch = (1, 32, 32)
weight = 2
lambda_pixel = 2
lr = 0.0002
beta1 = 0.5
beta2 = 0.999

loss_func_gan = nn.BCELoss().to(device)
loss_func_pix = nn.L1Loss().to(device)

gen_optim = optim.Adam(generator.parameters(), lr=lr, betas=(beta1,beta2))
dis_optim = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1,beta2))

def torch_to_numpy(tensor):
    img = np.transpose((tensor+1)/2, (1,2,0))
    plt.imshow(img)
    plt.axis('off')

def plot_generated_images(original, condition, generated, epoch, every):
    if epoch%every==0:
        for i in range(5):
            plt.figure(figsize=(12,4))
            plt.subplot(1,3,1)
            cond = condition[i].detach().cpu().numpy()
            torch_to_numpy(cond)
            plt.title('Condition Image')
            plt.subplot(1,3,2)
            img = original[i].detach().cpu().numpy()
            torch_to_numpy(img)
            plt.title('Original Image')
            plt.subplot(1,3,3)
            gen = generated[i].detach().cpu().numpy()
            torch_to_numpy(gen)
            plt.title('Generated Image')
            plt.show()

def train_discriminator(g_model, d_model,
                        real_A, real_B,
                        real_label, fake_label):
    dis_optim.zero_grad()
    
    fake_B = g_model(real_A)
    
    real_out = d_model(real_A, real_B)
    real_loss = torch.sum(loss_func_gan(real_out, real_label))
    
    fake_out = d_model(real_B, fake_B)
    fake_loss = torch.sum(loss_func_gan(fake_out, fake_label))
    
    d_loss = (real_loss + fake_loss) * weight
    d_loss.backward()
    dis_optim.step()

    return d_loss

def train_generator(g_model, d_model,
                    real_A, real_B,
                    real_label, fake_label):
    gen_optim.zero_grad()
    
    fake_B = g_model(real_A)
    
    dis_out = d_model(fake_B, real_B)
    
    gen_loss = loss_func_gan(dis_out, real_label)
    pix_loss = loss_func_pix(fake_B, real_B)
    
    g_loss = gen_loss + lambda_pixel * pix_loss
    g_loss.backward()
    gen_optim.step()

    return g_loss, fake_B

def train_gan(g_model, d_model,
              dataset,
              n_epochs,
              show_image_epoch,
              weight,
              lambda_pixel):
    g_model.train()
    d_model.train()

    g_loss_list, d_loss_list = [], []
    start_training = time.time()
    for epoch in tqdm(range(n_epochs)):
        init_time = time.time()
        batch_g_loss, batch_d_loss = 0, 0
        for batch, (img_A, img_B) in enumerate(dataset):
            real_A, real_B = img_A.to(device), img_B.to(device)

            real_label = torch.ones(img_A.size()[0], *patch, requires_grad=False).to(device)
            fake_label = torch.zeros(img_A.size()[0], *patch, requires_grad=False).to(device)

            """ train discriminator model """
            d_loss = train_discriminator(g_model, d_model,
                                         real_A, real_B,
                                         real_label, fake_label)
            batch_d_loss += d_loss.item()
            
            """ train generator model """
            g_loss, fake_B = train_generator(g_model, d_model,
                                     real_A, real_B,
                                     real_label, fake_label)
            batch_g_loss += g_loss.item()
            
        g_loss_list.append(batch_g_loss/(batch+1))
        d_loss_list.append(batch_d_loss/(batch+1))
        end_time = time.time()

        print(f'\nEpoch {epoch+1}/{n_epochs}'
              f'  [time: {end_time-init_time:.3f}s]\n'
              f'[generator loss: {batch_g_loss/(batch+1):.3f}]'
              f'  [discriminator loss: {batch_d_loss/(batch+1):.3f}]')
        
        plot_generated_images(real_A, real_B, fake_B, epoch+1, show_image_epoch)
        
    end_training = time.time()
    print(f'\nTotal time for training is {end_training-start_training:.3f}s')
    
    return {
        'real image': real_A,
        'fake image': fake_B,
        'generator loss': g_loss_list,
        'discriminator loss': d_loss_list,
    }