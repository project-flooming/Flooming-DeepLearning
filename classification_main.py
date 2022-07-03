import argparse
from msilib.schema import Class
from tabnanny import check

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset.classification_dataset import ClassificationDataset
from model.classification_model import VGG19
from train.classification_train import train_step


def get_args_parser():
    parser = argparse.ArgumentParser(description='Set Pix2Pix training', add_help=False)
    parser.add_argument('--path', default='C:/Users/user/MY_DL/flower/dataset/korean_flower/inputs', type=str,
                        help='Path of data')
    parser.add_argument('--img_size', default='256', type=int,
                        help='Input size of VGG19 model')
    parser.add_argument('--device', default='cuda', type=str,
                        help='Set device')
    parser.add_argument('--lr', default=0.0002, type=float,
                        help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float,
                        help='beta1 of Adam optimizer')
    parser.add_argument('--beta2', default=0.999, type=float,
                        help='beta2 of Adam optimizer')
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_classes', default=37, type=int)
    parser.add_argument('--checkpoint', default=True, type=bool)
    parser.add_argument('--earlystop', default=False, type=bool)
    parser.add_argument('--apply_scheduler', default=False, type=bool)
    return parser

def main(args):
    device = torch.device(args.device)

    path = args.path

    train_loader = DataLoader(
        ClassificationDataset(path=path, subset='train', img_size=args.img_size, transforms_=True),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    valid_loader = DataLoader(
        ClassificationDataset(path=path, subset='valid', img_size=args.img_size, transforms_=True),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    model = VGG19(num_classes=args.num_classes).to(device)
    print(model)

    history = train_step(
        model=model,
        train_data=train_loader,
        validation_data=valid_loader,
        epochs=args.epoch,
        learning_rate_scheduler=args.apply_scheduler,
        check_point=args.checkpoint,
        early_stop=args.earlystop,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser('VGG19 training', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)