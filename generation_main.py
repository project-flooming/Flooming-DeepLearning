import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset.generation_dataset import GenerationDataset
from model.generation_model import Generator, Discriminator
from train.generation_train import train_gan


def get_args_parser():
    parser = argparse.ArgumentParser(description='Set Pix2Pix training', add_help=False)
    parser.add_argument('--path', defaults='/dataset/', type=str,
                        help='Path of data')
    parser.add_argument('--img_size', default='256', type=int,
                        help='Input size of Pix2Pix model')
    parser.add_argument('--dropout', default=0.2, type=float,
                        help='The rate of dropout layer')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_availabel() else 'cpu', type=str,
                        help='Set device')
    parser.add_argument('--patch_size', default=(1, 32, 32), type=tuple,
                        help='Patch size for calculating loss')
    parser.add_argument('--weight', default=2, type=int,
                        help='Assign weight to the loss of discriminator model')
    parser.add_argument('--l1_weight', default=2., type=float,
                        help='Assign weight to the L1 Loss function')
    parser.add_argument('--lr', default=0.0002, type=float,
                        help='learning rate')
    parser.add_argument('--beta1', default=0.5, type=float,
                        help='beta1 of Adam optimizer')
    parser.add_argument('--beta2', default=0.999, type=float,
                        help='beta2 of Adam optimizer')
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--show_plot_epoch', default=10, type=int,
                        help='Show generated image')
    return parser

def main(args):
    device = torch.device(args.device)

    path = args.path

    data_loader = DataLoader(
        GenerationDataset(path=args.path, img_size=args.img_size, transforms_=True)
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    patch = args.patch_size
    weight = args.weight
    l1_weight = args.l1_weight
    lr = args.lr
    beta1 = args.beta1
    beta2 = args.beta2

    bce_loss = nn.BCELoss().to(device)
    l1_loss = nn.L1Loss().to(device)

    gen_optim = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    dis_optim = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

    history = train_gan(
        g_model=generator,
        d_model=discriminator,
        dataset=data_loader,
        n_epochs=args.epoch,
        show_image_epoch=args.show_plot_epoch,
        weight=weight,
        l1_weight=l1_weight,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pix2Pix training', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)