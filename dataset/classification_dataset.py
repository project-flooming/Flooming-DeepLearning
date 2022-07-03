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
        assert subset in ('train', 'valid')
        files0 = glob(path+'/'+subset+'/0/*.jpg')
        files1 = glob(path+'/'+subset+'/1/*.jpg')
        files2 = glob(path+'/'+subset+'/2/*.jpg')
        files3 = glob(path+'/'+subset+'/3/*.jpg')
        files4 = glob(path+'/'+subset+'/4/*.jpg')
        files5 = glob(path+'/'+subset+'/5/*.jpg')
        files6 = glob(path+'/'+subset+'/6/*.jpg')
        files7 = glob(path+'/'+subset+'/7/*.jpg')
        files8 = glob(path+'/'+subset+'/8/*.jpg')
        files9 = glob(path+'/'+subset+'/9/*.jpg')
        files10 = glob(path+'/'+subset+'/10/*.jpg')
        files11 = glob(path+'/'+subset+'/11/*.jpg')
        files12 = glob(path+'/'+subset+'/12/*.jpg')
        files13 = glob(path+'/'+subset+'/13/*.jpg')
        files14 = glob(path+'/'+subset+'/14/*.jpg')
        files15 = glob(path+'/'+subset+'/15/*.jpg')
        files16 = glob(path+'/'+subset+'/16/*.jpg')
        files17 = glob(path+'/'+subset+'/17/*.jpg')
        files18 = glob(path+'/'+subset+'/18/*.jpg')
        files19 = glob(path+'/'+subset+'/19/*.jpg')
        files20 = glob(path+'/'+subset+'/20/*.jpg')
        files21 = glob(path+'/'+subset+'/21/*.jpg')
        files22 = glob(path+'/'+subset+'/22/*.jpg')
        files23 = glob(path+'/'+subset+'/23/*.jpg')
        files24 = glob(path+'/'+subset+'/24/*.jpg')
        files25 = glob(path+'/'+subset+'/25/*.jpg')
        files26 = glob(path+'/'+subset+'/26/*.jpg')
        files27 = glob(path+'/'+subset+'/27/*.jpg')
        files28 = glob(path+'/'+subset+'/28/*.jpg')
        files29 = glob(path+'/'+subset+'/29/*.jpg')
        files30 = glob(path+'/'+subset+'/30/*.jpg')
        files31 = glob(path+'/'+subset+'/31/*.jpg')
        files32 = glob(path+'/'+subset+'/32/*.jpg')
        files33 = glob(path+'/'+subset+'/33/*.jpg')
        files34 = glob(path+'/'+subset+'/34/*.jpg')
        files35 = glob(path+'/'+subset+'/35/*.jpg')
        files36 = glob(path+'/'+subset+'/36/*.jpg')

        self.image_files = files0 + files1 + files2 + files3 + files4 + files5 + files6 + \
            files7 + files8 + files9 + files10 + files11 + files12 + files13 + files14 + \
            files15 + files16 + files17 + files18 + files19 + files20 + files21 + files22 + \
            files23 + files24 + files25 + files26 + files27 + files28 + files29 + files30 + \
            files31 + files32 + files33 + files34 + files35 + files36
        
        self.subset = subset
        self.transforms_ = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
        
        self.labels = [0]*len(files0) + [1]*len(files1) + [2]*len(files2) + [3]*len(files3) + \
            [4]*len(files4) + [5]*len(files5) + [6]*len(files6) + [7]*len(files7) + [8]*len(files8) + \
            [9]*len(files9) + [10]*len(files10) + [11]*len(files11) + [12]*len(files12) + [13]*len(files13) + \
            [14]*len(files14) + [15]*len(files15) + [16]*len(files16) + [17]*len(files17) + [18]*len(files18) + \
            [19]*len(files19) + [20]*len(files20) + [21]*len(files21) + [22]*len(files22) + [23]*len(files23) + \
            [24]*len(files24) + [25]*len(files25) + [26]*len(files26) + [27]*len(files27) + [28]*len(files28) + \
            [29]*len(files29) + [30]*len(files30) + [31]*len(files31) + [32]*len(files32) + [33]*len(files33) + \
            [34]*len(files34) + [35]*len(files35) + [36]*len(files36)
                    
        assert len(self.image_files) == len(self.labels), \
            f'The size of inputs and labels is different.'
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        images = Image.open(self.image_files[idx]).convert('RGB')
        labels = self.labels[idx]
        if self.transforms_ is not None:
            images = self.transforms_(images)
        return images, labels