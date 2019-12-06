import resnet
import torch
import torch.nn as nn
import torchvision
import xception

class Model_bach(nn.Module):
    def __init__(self, nb_classes=4):
        torch.nn.Module.__init__(self)
        base = torchvision.models.resnet34()
        #base = torchvision.models.resnet101(pretrained=True)
        #base = xception.xception(pretrained='imagenet')
       
        #self.base = nn.Sequential(*list(base.children())[:-1])
        self.base = base
        self.emb = nn.Linear(512, nb_classes)
        #self.emb = nn.Linear(2048, nb_classes)

    def forward(self, input):
        feat = self.base(input).squeeze()
        o = self.emb(feat)
        return o

