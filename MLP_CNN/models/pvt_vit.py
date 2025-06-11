import timm
import torch
from torch import nn, optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_vit_model(num_classes, in_channels):
    return timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=num_classes, in_chans=in_channels).to(device)

def get_pvt_model(num_classes, in_channels):
    return timm.create_model('pvt_v2_b0', pretrained=False, num_classes=num_classes, in_chans=in_channels).to(device)