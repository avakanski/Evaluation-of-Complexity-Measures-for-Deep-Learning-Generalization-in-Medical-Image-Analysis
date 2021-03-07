# This file is used for defining the models

# SingleTask model is a VGG-like network with blocks of 2 convolutional layers and a maxpooling layer
# MultiTask model applies both classification and segmentation, by adding a U-Net-like decodes to the SingleTask model

import torch
import torch.nn as nn
import torch.nn.functional as F

class Network_SingleTask(nn.Module):
    def __init__(self, n_channels, n_classes, batch_norm=True, dropout=True):
        super(Network_SingleTask, self).__init__()
        self.batch_norm = batch_norm
        self.dropout = dropout
        
        # Encoder
        self.Conv_Block_1 = DoubleConv(n_channels, 32, batch_norm)
        self.Conv_Block_2 = DownConv(32, 64, batch_norm)
        self.Conv_Block_3 = DownConv(64, 128, batch_norm)
        self.Conv_Block_4 = DownConv(128, 256, batch_norm)
        self.Conv_Block_5 = DownConv(256, 512, batch_norm)
        
        # Classification head
        self.Classification_Head = ClassifyLayers(512, 256, 64, n_classes, dropout)

    def forward(self, x):
        # Encoder
        x1 = self.Conv_Block_1(x)
        x2 = self.Conv_Block_2(x1)
        x3 = self.Conv_Block_3(x2)
        x4 = self.Conv_Block_4(x3)
        x5 = self.Conv_Block_5(x4)

        # Average pooling
        x_ap = x5.mean(dim=(2, 3))
        
        # Classification head
        x_class = self.Classification_Head(x_ap)
        return x_class

 
# From: https://github.com/milesial/Pytorch-UNet
class Network_MultiTask(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(Network_MultiTask, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Encoder
        self.Conv_Block_1 = DoubleConv(n_channels, 32)
        self.Conv_Block_2 = DownConv(32, 64)
        self.Conv_Block_3 = DownConv(64, 128)
        self.Conv_Block_4 = DownConv(128, 256)
        self.Conv_Block_5 = DownConv(256, 512)
        
        # Decoder
        self.UpConv_Block_4 = UpConv(512, 256)
        self.UpConv_Block_3 = UpConv(256, 128)
        self.UpConv_Block_2 = UpConv(128, 64)
        self.UpConv_Block_1 = UpConv(64, 32)
        self.Out_Last = ConvLast(32)
        
        # Classification head
        self.Classification_Head = ClassifyLayers(512, 256, 64, n_classes)
        
    def forward(self, x):
        # Encoder
        x1 = self.Conv_Block_1(x)
        x2 = self.Conv_Block_2(x1)
        x3 = self.Conv_Block_3(x2)
        x4 = self.Conv_Block_4(x3)
        x5 = self.Conv_Block_5(x4)
        # Decoder
        x = self.UpConv_Block_4(x5, x4)
        x = self.UpConv_Block_3(x, x3)
        x = self.UpConv_Block_2(x, x2)
        x = self.UpConv_Block_1(x, x1)
  
        # Segmentation branch output
        pred_mask = torch.sigmoid(self.Out_Last(x))
        
        # Average pooling
        avg_pool = x5.mean(dim=(2, 3))

        # Classification branch output   
        pred_class = self.Classification_Head(avg_pool)

        # Combined output
        return [pred_class, pred_mask]

 
class DoubleConv(nn.Module):
    # Two convolutional layers: (in_channel size, out_channels size, kernel_size, stride, padding)
    def __init__(self, in_channels, out_channels, batch_norm=True):
        super().__init__()
        
        block = []

        block.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        if batch_norm:
            block.append(nn.BatchNorm2d(out_channels))
        block.append(nn.ReLU(inplace=True))
        
        block.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        if batch_norm:
            block.append(nn.BatchNorm2d(out_channels))
        block.append(nn.ReLU(inplace=True))
        
        self.block = nn.Sequential(*block)
    def forward(self, x):
        return self.block(x)


class DownConv(nn.Module):
    # Downscaling with maxpooling, then double convolution
    def __init__(self, in_channels, out_channels, batch_norm=True):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            # max pooling (kernel_size, stride)
            nn.MaxPool2d(2, 2),
            DoubleConv(in_channels, out_channels, batch_norm)
        )
    def forward(self, x):
        return self.maxpool_conv(x)


class UpConv(nn.Module):
    # Upscaling, then double convovolution, used for the decoder in the segmentation branch
    def __init__(self, in_channels, out_channels, batch_norm=True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, batch_norm)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ConvLast(nn.Module):
    # Last convolutional layer for the segmentation branch
    def __init__(self, in_channels):
        super(ConvLast, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
    def forward(self, x):
        return self.conv(x)

        
class ClassifyLayers(nn.Module):
    # Classification head consisting of 3 fully-connected layers
    def __init__(self, in_channels, mid_channels1, mid_channels2, n_classes, dropout=True):
        super(ClassifyLayers, self).__init__()
        
        dense_layers = []
        dense_layers.append(nn.Linear(in_channels, mid_channels1))
        dense_layers.append(nn.ReLU(inplace=True))
        if dropout:
            dense_layers.append(nn.Dropout(0.25))
        dense_layers.append(nn.Linear(mid_channels1, mid_channels2))
        dense_layers.append(nn.ReLU(inplace=True))
        if dropout:
            dense_layers.append(nn.Dropout(0.25))
        dense_layers.append(nn.Linear(mid_channels2, n_classes))
        
        self.dense_layers = nn.Sequential(*dense_layers)
        
    def forward(self, x):
        return self.dense_layers(x)
