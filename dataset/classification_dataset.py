from glob import glob
from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

"""
path : dataset/
├── images
│    ├─ class 1
│        ├─ img1.jpg
│        ├─ ...
│    ├─ class 2
│        ├─ img1.jpg
│        ├─ ...
│    ├─ class 3
│        ├─ img1.jpg
│        ├─ ...
│    ├─ class 4
│        ├─ img1.jpg
│        ├─ ...
│    ├─ ...
│        ├─ ...
│        ├─ ...
"""

class ClassificationDataset(Dataset):
    def __init__(self, path, subset, img_size=256, transforms_=None):
        assert subset in ('train', 'valid', 'test')
        image1_files = glob(path+'/inputs/'+subset+'*.jpg')
        image2_files = glob(path+'/inputs/'+subset+'*.jpg')
        image3_files = glob(path+'/inputs/'+subset+'*.jpg')
        image4_files = glob(path+'/inputs/'+subset+'*.jpg')
        image5_files = glob(path+'/inputs/'+subset+'*.jpg')
        image6_files = glob(path+'/inputs/'+subset+'*.jpg')
        image7_files = glob(path+'/inputs/'+subset+'*.jpg')
        image8_files = glob(path+'/inputs/'+subset+'*.jpg')
        image9_files = glob(path+'/inputs/'+subset+'*.jpg')
        image10_files = glob(path+'/inputs/'+subset+'*.jpg')
        
        self.image_files = image1_files + image2_files + image3_files + \
                            image4_files + image5_files + image6_files + \
                            image7_files + image8_files + image9_files + image10_files
        self.subset = subset
        self.transforms_ = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        
        self.labels = [0]*len(image1_files) + [1]*len(image2_files) + \
                        [2]*len(image3_files) + [3]*len(image4_files) + \
                        [4]*len(image5_files) + [5]*len(image5_files) + \
                        [6]*len(image3_files) + [7]*len(image4_files) + \
                        [8]*len(image5_files) + [9]*len(image5_files) + \
                        [10]*len(image3_files)
        
        assert len(self.image_files) == len(self.labels), \
            f'The size of inputs and labels is different.'
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        images = Image.open(self.image_files[idx])
        labels = self.labels[idx]
        if self.transforms_ is not None:
            images = self.transforms_(images)
        return images, labels