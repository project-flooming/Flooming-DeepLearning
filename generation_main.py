import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset.generation_dataset import GenerationDataset
from model.pix2pix import Generator, Discriminator
from train.generation_train import train_gan


def get_args_parser():
    parser = argparse.ArgumentParser(description='Set Pix2Pix training', add_help=False)
    parser.add_argument('--path', defaults='/dataset/', type=str,
                        help='Path of data')
    parser.add_argument('--img_size', default='256', type=int,
                        help='Input size of Pix2Pix model')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_availabel() else 'cpu', type=str,
                        help='Set device')
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--show_plot_epoch', default=10, type=int,
                        help='Show generated image')
    parser.add_argument('--weight', default=20, type=int,
                        help='weight of discriminator loss')
    parser.add_argument('--l1_weight', default=30, type=int,
                        help='weight of lambda pixel')
    return parser

def main(args):
    device = torch.device(args.device)

    path = args.path

    data_loader = DataLoader(
        GenerationDataset(path=args.path, img_size=args.img_size, transforms_=True),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    history = train_gan(
        g_model=generator,
        d_model=discriminator,
        data_loader=data_loader,
        n_epochs=args.epoch,
        show_image_epoch=args.show_plot_epoch,
        weight=args.weight,
        l1_weight=args.l1_weight,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pix2Pix training', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)