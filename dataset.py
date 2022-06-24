from glob import glob
from PIL import Image

import torch
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

class FloomingDataset(Dataset):
    def __init__(self, path, transforms_=None):
        self.input_files = glob(path+'/inputs/*.jpg')
        self.label_files = glob(path+'/labels/*.jpg')
        self.transforms_ = transforms_
        assert len(self.input_files) == len(self.label_files), \
            f'The size of inputs and labels is different.'
        
    def __len__(self):
        return len(self.input_files)
    
    def __getitem__(self, idx):
        inputs = Image.open(self.input_files[idx])
        labels = Image.open(self.label_files[idx])
        if self.transforms_ is not None:
            inputs = self.transforms_(inputs)
            labels = self.transforms_(labels)
        return inputs, labels