import torch
from torchvision import models
from collections import namedtuple


class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained = models.vgg16(
            weights=models.VGG16_Weights.IMAGENET1K_V1
        ).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        VggOutputs = namedtuple(
            "VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"]
        )
        out = VggOutputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out
