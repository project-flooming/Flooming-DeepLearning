import torch
import torch.nn as nn
import torch.nn.init as init

class UNetEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, normalize=True, dropout=0.2):
        super(UNetEncoder, self).__init__()
        layers = [
            nn.Conv2d(in_dim, out_dim, kernel_size=4, stride=2, padding=1),
        ]
        if normalize:
            layers.append(nn.BatchNorm2d(out_dim))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.encoder(x)
    
class UNetDecoder(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2):
        super(UNetDecoder, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, x, skip):
        x = self.decoder(x)
        x = torch.cat([x, skip], dim=1)
        return x
    
class Generator(nn.Module):
    def __init__(self, in_dim=3, num_filters=64, out_dim=3):
        super(Generator, self).__init__()
        self.encoder1 = UNetEncoder(in_dim, num_filters, normalize=False)
        self.encoder2 = UNetEncoder(num_filters, num_filters*2)
        self.encoder3 = UNetEncoder(num_filters*2, num_filters*4)
        self.encoder4 = UNetEncoder(num_filters*4, num_filters*8, dropout=0.2)
        self.encoder5 = UNetEncoder(num_filters*8, num_filters*8, dropout=0.2)
        self.encoder6 = UNetEncoder(num_filters*8, num_filters*8, dropout=0.2)
        self.encoder7 = UNetEncoder(num_filters*8, num_filters*8, dropout=0.2)
        self.encoder8 = UNetEncoder(num_filters*8, num_filters*8, normalize=False, dropout=0.2)
        
        self.decoder1 = UNetDecoder(num_filters*8, num_filters*8, dropout=0.2)
        self.decoder2 = UNetDecoder(num_filters*8 + num_filters*8, num_filters*8, dropout=0.2)
        self.decoder3 = UNetDecoder(num_filters*8 + num_filters*8, num_filters*8, dropout=0.2)
        self.decoder4 = UNetDecoder(num_filters*8 + num_filters*8, num_filters*8, dropout=0.2)
        self.decoder5 = UNetDecoder(num_filters*8 + num_filters*8, num_filters*4, dropout=0.2)
        self.decoder6 = UNetDecoder(num_filters*8, num_filters*2)
        self.decoder7 = UNetDecoder(num_filters*4, num_filters)
        self.decoder8 = nn.Sequential(
            nn.ConvTranspose2d(num_filters*2, out_dim, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )
        self._init_weight_()
        
    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        e6 = self.encoder6(e5)
        e7 = self.encoder7(e6)
        e8 = self.encoder8(e7)
        
        d1 = self.decoder1(e8, e7)
        d2 = self.decoder2(d1, e6)
        d3 = self.decoder3(d2, e5)
        d4 = self.decoder4(d3, e4)
        d5 = self.decoder5(d4, e3)
        d6 = self.decoder6(d5, e2)
        d7 = self.decoder7(d6, e1)
        output = self.decoder8(d7)
        return output
    
    def _init_weight_(self, mean=0.0, std=0.02):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal(m.weight, mean, std)
            if isinstance(m, nn.ConvTranspose2d):
                init.normal(m.weight, mean, std)
    
class ConvBNReLU(nn.Module):
    def __init__(self, in_dim, out_dim, normalize=True):
        super(ConvBNReLU, self).__init__()
        layers = [
            nn.Conv2d(in_dim, out_dim, kernel_size=4, stride=2, padding=1)
        ]
        if normalize:
            layers.append(nn.BatchNorm2d(out_dim))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)
    
class Discriminator(nn.Module):
    def __init__(self, in_dim=3, num_filters=64):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            ConvBNReLU(in_dim*2, num_filters, normalize=False),
            ConvBNReLU(num_filters, num_filters*2),
            ConvBNReLU(num_filters*2, num_filters*4),
            ConvBNReLU(num_filters*4, num_filters*8),
            nn.ZeroPad2d((1,0,1,0)),
            nn.Conv2d(num_filters*8, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid(),
        )
        self._init_weight_()
        
    def forward(self, imgA, imgB):
        inputs = torch.cat([imgA, imgB], dim=1)
        output = self.discriminator(inputs)
        return output
    
    def _init_weight_(self, mean=0.0, std=0.02):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal(m.weight, mean, std)