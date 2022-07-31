import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import make_grid

from ..model.pix2pix import Generator, Discriminator

device = torch.device('cuda')
patch = (1, 32, 32)
lr = 0.0002
beta1 = 0.5
beta2 = 0.999

device = torch.device('cuda')
g_model = Generator().to(device)
d_model = Discriminator().to(device)
bce_loss = nn.BCELoss().to(device)
l1_loss = nn.L1Loss().to(device) 

g_optim = optim.Adam(g_model.parameters(), lr=lr, betas=(beta1, beta2))
d_optim = optim.Adam(d_model.parameters(), lr=lr, betas=(beta1, beta2))

def show_generated_image(original, condition, generated, epoch, every):
    if epoch % every == 0: 
        fig, axes = plt.subplots(3,1, figsize=(20,15))
        cond = (make_grid(condition[:5], nrow=5, padding=3, normalize=True).cpu()).permute(1,2,0)
        axes[0].imshow(cond)
        axes[0].axis('off')
        axes[0].set_title('Condition Image', fontsize=25)
        img = (make_grid(original[:5], nrow=5, padding=3, normalize=True).cpu()).permute(1,2,0)
        axes[1].imshow(img)
        axes[1].axis('off')
        axes[1].set_title('Original Image', fontsize=25)
        gen = (make_grid(generated[:5], nrow=5, padding=3, normalize=True).cpu()).permute(1,2,0)
        axes[2].imshow(gen)
        axes[2].axis('off')
        axes[2].set_title('Generated Image', fontsize=25)
        plt.show()

def train_gan(
    g_model, d_model,
    data_loader,
    n_epochs,
    show_image_epoch,
    weight,
    l1_weight,
):
    g_model.train()
    d_model.train()

    d_loss_list, g_loss_list = [], []
    starting = time.time()
    for epoch in tqdm(range(n_epochs)):
        init_time = time.time()
        batch_g_loss, batch_d_loss = 0, 0
        for batch, [img_A, img_B] in enumerate(data_loader):
            real_A, real_B = img_A.to(device), img_B.to(device)

            real_label = torch.ones(img_A.size()[0], *patch, requires_grad=False).to(device)
            fake_label = torch.zeros(img_A.size()[0], *patch, requires_grad=False).to(device)

            """ train discriminator model """
            d_optim.zero_grad()

            fake_B = g_model(real_A)

            real_out = d_model(real_A, real_B)
            real_loss = torch.sum(bce_loss(real_out, real_label))

            fake_out = d_model(real_B, fake_B)
            fake_loss = torch.sum(bce_loss(fake_out, fake_label))

            d_loss = (real_loss + fake_loss) * weight
            d_loss.backward(retain_graph=True)
            d_optim.step()
            batch_d_loss += d_loss.item()

            """ train generator model """
            g_optim.zero_grad()

            out_dis = d_model(fake_B, real_B)

            gen_loss = bce_loss(out_dis, real_label)
            pix_loss = l1_loss(fake_B, real_B)
            
            g_loss = gen_loss + pix_loss * l1_weight
            g_loss.backward()
            g_optim.step()
            batch_g_loss += g_loss.item()

        d_loss_list.append(batch_d_loss/(batch+1))
        g_loss_list.append(batch_g_loss/(batch+1))
        end_time = time.time()

        torch.save(g_model.state_dict(), f'./weight/generator_weight_{epoch+1}.pt')
        torch.save(d_model.state_dict(), f'./weight/discriminator_weight_{epoch+1}.pt')

        print(f'\nEpoch {epoch+1}/{n_epochs}'
              f'  [time: {end_time-init_time:.3f}s]'
              f'  [generator loss: {batch_g_loss/(batch+1):.3f}]'
              f'  [discriminator loss: {batch_d_loss/(batch+1):.3f}]')

        show_generated_image(real_A, real_B, fake_B, epoch+1, show_image_epoch)
    
    ending = time.time()
    print(f'\nTotal time for training is {ending-starting:.3f}s')

    return {
        'real image': real_A,
        'fake image': fake_B,
        'generator loss': g_loss_list,
        'discriminator loss': d_loss_list,
    }