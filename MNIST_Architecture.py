import torch
import torch.nn as nn
from torch.autograd import Variable
from Capsule_Layers import *


class Net(nn.Module):
   
   def __init__(self,input_shape,args):
       super(Net, self).__init__()
       dim_caps = int(args.dimcaps)
       self.layernum = int(args.layernum)
      

       self.BN1 = Conv2d_bn(input_shape = input_shape,filters=64,kernel_size=3, strides=1, padding='same')
       self.BN2 = Conv2d_bn(input_shape = [None,64,28,28],filters=64,kernel_size=3, strides=1, padding='same')

       ## Primary Capsules
       self.Primary_Cap = PrimaryCap( input_shape = [None,64, 28,28], n_channels = 8, dim_capsule = dim_caps, decrease_resolution = True)

       ## Convolutional Capsules
       if self.layernum == 0:
         self.FullyConvCaps = FullyConvCaps(input_shape = [None, 8, 16, 14, 14],n_channels = 10,dim_capsule=dim_caps)
        
       elif self.layernum == 1:
         self.ConvCaps = ConvCaps(input_shape = [None, 8, 16, 14, 14],n_channels=8, dim_capsule=dim_caps, decrease_resolution = True)
         self.FullyConvCaps = FullyConvCaps(input_shape = [None, 8, 16, 7, 7],n_channels = 10,dim_capsule=dim_caps)

       elif self.layernum == 2:
         self.ConvCaps_2 = Add_ConvCaps(input_shape = [None, 8, 16, 14, 14], layer_num = 2,dim_capsule = dim_caps)
         self.FullyConvCaps = FullyConvCaps(input_shape = [None, 8, 16, 7, 7],n_channels = 10,dim_capsule=dim_caps)

       elif self.layernum == 3:
         self.ConvCaps_3 = Add_ConvCaps(input_shape = [None, 8, 16, 14, 14], layer_num = 3,dim_capsule = dim_caps)
         self.FullyConvCaps = FullyConvCaps(input_shape = [None, 8, 16, 7, 7],n_channels = 10,dim_capsule=dim_caps)

        
        
       self.decoder = Decoder(input_size = input_shape[1:])
       self.tanh = nn.Tanh()
   
   def forward(self, inputs):

     bn1 = self.tanh(self.BN1 (inputs))
     bn2 = self.tanh(self.BN2 (bn1))

     primary_cap= self.tanh(self.Primary_Cap(bn2))

      ## Convolutional Capsules
     if self.layernum == 0:
       FullyConvCaps = self.tanh(self.FullyConvCaps(primary_cap))
       return FullyConvCaps
         
     elif self.layernum == 1:
       ConvCaps = self.tanh(self.ConvCaps(primary_cap))
       FullyConvCaps = self.tanh(self.FullyConvCaps(ConvCaps))
       return FullyConvCaps

     elif self.layernum == 2:
       ConvCaps = self.ConvCaps_2(primary_cap)
       FullyConvCaps = self.tanh(self.FullyConvCaps(ConvCaps))
       return FullyConvCaps
     
     elif self.layernum == 3:
       ConvCaps = self.ConvCaps_3(primary_cap)
       FullyConvCaps = self.tanh(self.FullyConvCaps(ConvCaps))
       return FullyConvCaps

   def caps_loss(self,y_true, y_pred, x, lam_recon):
    """
    Capsule loss = Margin loss + lam_recon * reconstruction loss.
    :param y_true: true labels, one-hot coding, size=[batch, classes]
    :param y_pred: predicted labels by CapsNet, size=[batch, classes]
    :param x: input data, size=[batch, channels, width, height]
  
    :param lam_recon: coefficient for reconstruction loss
    :return: Variable contains a scalar loss value.
    """
    # x_recon: reconstructed data, size is same as `x`
    x_recon = self.decoder(y_pred, y_true)
    y_pred = y_pred.norm(dim=-1)
    L = y_true * torch.clamp(0.9 - y_pred, min=0.) ** 2 + \
        0.5 * (1 - y_true) * torch.clamp(y_pred - 0.1, min=0.) ** 2
    L_margin = L.sum(dim=1).mean()

    L_recon = nn.MSELoss()(x_recon, x)

    return L_margin + lam_recon * L_recon




