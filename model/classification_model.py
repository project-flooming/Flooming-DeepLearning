import torch 
import torch.nn as nn
import torchvision.models as models

class VGG19(nn.Module):
    def __init__(
        self,
        in_dim=3,
        num_filters=64,
        num_classes=1000,
        pre_trained=True,
    ):
        super(VGG19, self).__init__()
        pre_trained = models.vgg19_bn(pretrained=pre_trained)
        self.features = pre_trained.features
        self.gap = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x