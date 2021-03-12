#-----------------------------------------------------------------------#
#                          Library imports                              #
#-----------------------------------------------------------------------#
import torch
import torch.nn as nn
from collections import OrderedDict


#-----------------------------------------------------------------------#
#                             UNet_2D                                   #
#        2 Dimensional Implementation of UNet acrchitecture             #
#-----------------------------------------------------------------------#
# Reference:                                                            #
# Ronneberger, O.; Fischer, P.; Brox, T. U-Net: Convolutional Networks  #
# for Biomedical Image Segmentation. In Proceedings of the International#
# Conference on Medical image computing and computer-assisted           #
# intervention, Munich, Germany, 5–9 October 2015; pp. 234–241.         #
#-----------------------------------------------------------------------#
# Adopted from:                                                         #
# The net:                                                              #
# github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/unet.py #                                                 
# https://github.com/hlamba28/UNET-TGS/blob/master/TGS%20UNET.ipynb     #
# The weight initialization:                                            #
# https://discuss.pytorch.org/t/pytorch-how-to-initialize-weights       #
#   /81511/4                                                            #
# https://discuss.pytorch.org/t                                         #
#   /how-are-layer-weights-and-biases-initialized-by-default/13073/24   #
# The weight standardization:                                           #
# https://github.com/joe-siyuan-qiao/WeightStandardization              #
# Siyuan Qiao et al., Micro-Batch Training with Batch-Channel           #
#      Normalization and Weight Standardization,arXiv:1903.10520v2,2020 #
#-----------------------------------------------------------------------#
# in_channels:   number of input channels                               #
# out_channels:  number of output channels                              #
# init_features: number of filters in the first encoding layer, it      #
#                doubles at the successive encoding steps and halves at #
#                each decoding layer.                                   #
# dropout_p:     dropout probability                                    #
# mean, std:     mean and standard deviation to be used for weight      #
#                initialization using Gaussian distribution.            #
#                The standard deviation is the square root of (2/N),    #
#                where N is the number of incoming nodes of one neuron. #   
#-----------------------------------------------------------------------#
class UNet_2D(nn.Module):
    #2D UNet architecture
    def __init__(self, in_channels=1, out_channels=1, init_features=64, dropout_p= 0.5):
        super().__init__()
        features = init_features
        
        # Encoding layers
        self.encoder1 = UNet_2D._block(in_channels, features)   
        self.encoder2 = UNet_2D._block(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet_2D._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet_2D._block(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck layer
        self.bottleneck = UNet_2D._block(features * 8, features * 16)

        # Decoding layers
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet_2D._block((features * 8) * 2, features * 8)
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet_2D._block((features * 4) * 2, features * 4)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet_2D._block((features * 2) * 2, features * 2)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet_2D._block(features * 2, features)

        # output layer
        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

        # Max Pool
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout
        self.dropout = nn.Dropout(dropout_p)

        self.weight_init()   

    @staticmethod
    def normal_init(m, mean, std):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            m.weight.data.normal_(mean, std)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.weight.data.fill_(1)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    
    # Weight initialization 
    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    mean = 0
                    # standard deviation based on a 3*3 convolution
                    std =  (2/(3*3* m.out_channels))**(0.5)
                    normal_init(m, self.mean, self.std)
            except:
                pass
            
    # Weight standardization: A normalization to be used with group normalization (micro_batch)
    def WS(self):
        for block in self._modules:
            if isinstance(block, nn.MaxPool2d) or isinstance(block, nn.ConvTranspose2d):
                pass
            else:
                for m in block:
                    if isinstance(m, nn.Conv2d):
                        weight = m.weight
                        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                      keepdim=True).mean(dim=3, keepdim=True)
                        weight = weight - weight_mean
                        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
                        weight = weight / std.expand_as(weight)
                        m.weight.data = weight
          

    def forward(self, x):
        # Encoding path
        enc1 = self.encoder1(x)
        p1 = self.dropout(self.pool(enc1))
        enc2 = self.encoder2(p1)
        p2 = self.dropout(self.pool(enc2))
        enc3 = self.encoder3(p2)
        p3 = self.dropout(self.pool(enc3))
        enc4 = self.encoder4(p3)
        p4 = self.dropout(self.pool(enc4))

        # Bottleneck
        bottleneck = self.bottleneck(p4)      

        # Decoding path
        dec4 = self.dropout(self.upconv4(bottleneck))
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec4 = self.dropout(self.upconv4(bottleneck))
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.dropout(self.upconv3(dec4))
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.dropout(self.upconv2(dec3))
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.dropout(self.upconv1(dec2))
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        #self.WS()
        # Output
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_features, out_features):               
        return nn.Sequential(OrderedDict([
                    ("conv1",nn.Conv2d(
                        in_channels=in_features,
                        out_channels=out_features,
                        kernel_size=3,
                        padding=1,
                        bias=False)),
                    ("norm1", nn.BatchNorm2d(num_features=out_features)),
                    #("relu1", nn.ReLU(inplace=True)),
                    ("swish1", nn.SiLU(inplace=True)),
                    ("conv2",nn.Conv2d(
                        in_channels=out_features,
                        out_channels=out_features,
                        kernel_size=3,
                        padding=1,
                        bias=False)),
                    ("norm2", nn.BatchNorm2d(num_features=out_features)),
                    #("relu2", nn.ReLU(inplace=True))
                    ("swish2", nn.SiLU(inplace=True))
                ]))


#-----------------------------------------------------------------------#
#                             UNet_3D                                   #
#        3 Dimensional Implementation of UNet acrchitecture             #
#-----------------------------------------------------------------------#
# Reference:                                                            #
# Çiçek, Ö. et al. 3D U-Net: Learning dense volumetric segmentation     #
# from sparse annotation. In Proceedings of the International Conference#
# on Medical Image Computing and Computer-Assisted Intervention,        #
# Athens, Greece, 17–21 October 2016; pp. 424–432.                      #
#-----------------------------------------------------------------------#
# Adopted from:                                                         #
# The net:                                                              #
# github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/unet.py #                                                 
# https://github.com/hlamba28/UNET-TGS/blob/master/TGS%20UNET.ipynb     #
# The weight initialization:                                            #
# https://discuss.pytorch.org/t/pytorch-how-to-initialize-weights       #
#   /81511/4                                                            #
# https://discuss.pytorch.org/t                                         #
#   /how-are-layer-weights-and-biases-initialized-by-default/13073/24   #
# The weight standardization:                                           #
# https://github.com/joe-siyuan-qiao/WeightStandardization              #
# Siyuan Qiao et al., Micro-Batch Training with Batch-Channel           #
#      Normalization and Weight Standardization,arXiv:1903.10520v2,2020 #
#-----------------------------------------------------------------------#
# in_channels:   number of input channels                               #
# out_channels:  number of output channels                              #
# init_features: number of filters in the first encoding layer, it      #
#                doubles at the successive encoding steps and halves at #
#                each decoding layer.                                   #
# dropout_p:     dropout probability                                    #
# mean, std:     mean and standard deviation to be used for weight      #
#                initialization using Gaussian distribution.            #
#                The standard deviation is the square root of (2/N),    #
#                where N is the number of incoming nodes of one neuron. #   
#-----------------------------------------------------------------------#

class UNet_3D(nn.Module):
    #3D UNet architecture
    def __init__(self, in_channels=1, out_channels=1, init_features=64, dropout_p= 0.5):
        super().__init__()
        features = init_features
        
        # Encoding layers
        self.encoder1 = UNet_3D._block(in_channels, features)   
        self.encoder2 = UNet_3D._block(features, features * 2)
        self.encoder3 = UNet_3D._block(features * 2, features * 4)
        self.encoder4 = UNet_3D._block(features * 4, features * 8)

        # Bottleneck layer
        self.bottleneck = UNet_3D._block(features * 8, features * 16)

        # Decoding layers
        self.upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet_3D._block((features * 8) * 2, features * 8)
        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet_3D._block((features * 4) * 2, features * 4)
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet_3D._block((features * 2) * 2, features * 2)
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet_3D._block(features * 2, features)

        # output layer
        self.conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

        # Max Pool
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Dropout
        self.dropout = nn.Dropout(dropout_p)

        self.weight_init()   
    
    @staticmethod
    def normal_init(m, mean, std):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            m.weight.data.normal_(mean, std)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.BatchNorm1d)):
            m.weight.data.fill_(1)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    # Weight initialization                   
    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    mean = 0
                    # standard deviation based on a 3*3 convolution
                    std =  (2/(3*3*3* m.out_channels))**(0.5)
                    normal_init(m, mean, std)
            except:
                pass
    
    # Weight standardization:A normalization to be used with group normalization (micro_batch)
    def WS(self):
        for block in self._modules:
            if isinstance(block, nn.MaxPool2d) or isinstance(block, nn.ConvTranspose2d):
                pass
            else:
                for m in block:
                    if isinstance(m, nn.Conv2d):
                        #ref:https://github.com/joe-siyuan-qiao/WeightStandardization
                        weight = m.weight
                        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                      keepdim=True).mean(dim=3, keepdim=True)
                        weight = weight - weight_mean
                        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
                        weight = weight / std.expand_as(weight)
                        m.weight.data = weight
    

    def forward(self, x):
        x = x.squeeze(0)
        # Encoding path
        enc1 = self.encoder1(x)
        p1 = self.dropout(self.pool(enc1))
        enc2 = self.encoder2(p1)
        p2 = self.dropout(self.pool(enc2))
        enc3 = self.encoder3(p2)
        p3 = self.dropout(self.pool(enc3))
        enc4 = self.encoder4(p3)
        p4 = self.dropout(self.pool(enc4))

        # Bottleneck
        bottleneck = self.bottleneck(p4)      

        # Decoding path
        dec4 = self.dropout(self.upconv4(bottleneck))
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec4 = self.dropout(self.upconv4(bottleneck))
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.dropout(self.upconv3(dec4))
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.dropout(self.upconv2(dec3))
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.dropout(self.upconv1(dec2))
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        #self.WS()
        # Output
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_features, out_features):               
        return nn.Sequential(OrderedDict([
                    ("conv1",nn.Conv3d(
                        in_channels=in_features,
                        out_channels=out_features,
                        kernel_size=3,
                        padding=1,
                        bias=False)),
                    ("norm1", nn.BatchNorm3d(num_features=out_features)),
                    ("relu1", nn.ReLU(inplace=True)),
                    #("swish1", nn.SiLU(inplace=True)),
                    ("conv2",nn.Conv3d(
                        in_channels=out_features,
                        out_channels=out_features,
                        kernel_size=3,
                        padding=1,
                        bias=False)),
                    ("norm2", nn.BatchNorm3d(num_features=out_features)),
                    ("relu2", nn.ReLU(inplace=True))
                    #("swish2", nn.SiLU(inplace=True))
                ]))

