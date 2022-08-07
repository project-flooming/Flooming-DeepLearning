from glob import glob
from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

"""
path : dataset/
├── inputs
│    ├─ input_image1.jpg
│    ├─ ...
├── labels
│    ├─ label_image1.jpg
│    ├─ ...
"""

class GenerationDataset(Dataset):
    def __init__(self, path, img_size, transforms_=None):
        self.input_files = glob(path+'/inputs/*.jpg')
        self.label_files = glob(path+'/labels/*.jpg')
        assert len(self.input_files) == len(self.label_files), \
            f'The size of inputs and labels is different.'

        self.transforms_ = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ])
        
    def __len__(self):
        return len(self.input_files)
    
    def __getitem__(self, idx):
        inputs = Image.open(self.input_files[idx])
        labels = Image.open(self.label_files[idx])
        if self.transforms_ is not None:
            inputs = self.transforms_(inputs)
            labels = self.transforms_(labels)
        return inputs, labels