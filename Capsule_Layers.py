import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class Conv2d_bn(nn.Module):
    def __init__(self,input_shape, filters, kernel_size=3, strides=1, padding='same', activation='relu'):
      super(Conv2d_bn, self).__init__()
      if padding == 'same':
        padding = int((kernel_size - 1)/2)
      else:
        padding = 0
      input_channels = int(input_shape[1])
      #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      self.layer = nn.Sequential()
      self.layer.add_module("Conv1",nn.Conv2d(in_channels=input_channels,out_channels=filters,kernel_size=kernel_size,padding=padding))
      self.layer.add_module("BN1", nn.BatchNorm2d(num_features=filters, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
      self.layer.add_module("Relu1", nn.ReLU())

    def forward(self, x):
      x = self.layer(x)
      return x

class Decoder(nn.Module):
    def __init__(self,input_size):
        super(Decoder, self).__init__()
        self.input_size= input_size
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.input_size = self.input_size.to(device)
        self.layer = nn.Sequential(
            nn.Linear(16*10, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.input_size[0] * self.input_size[1] * self.input_size[2]),
            nn.Sigmoid()
        )
    
    def forward(self,inputs,y=None):

        length = inputs.norm(dim=-1)

        if y is None:  # during testing, no label given. create one-hot coding using `length`
            index = length.max(dim=1)[1]
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            y = Variable(torch.zeros(length.size()).scatter_(1, index.view(-1, 1).cpu().data, 1.).to(device))          
        # print(inputs.shape, y[:,:,None].shape)
        reconstruction = self.layer((inputs * y[:, :, None]).reshape(inputs.size(0), -1))
        return reconstruction.view(-1, *self.input_size)


class PrimaryCap(nn.Module):
    def __init__(self,input_shape, n_channels, dim_capsule, decrease_resolution=False):

      super(PrimaryCap, self).__init__()
      self.dim_capsule = dim_capsule
      self.n_channels = n_channels
      if decrease_resolution == True:
          self.stride = 2
      else:
          self.stride = 1
      self.input_height = input_shape[2]
      self.input_width = input_shape[3]
      self.input_num_features = input_shape[1]

      w = torch.empty(self.dim_capsule*self.n_channels,self.input_num_features,3, 3)
      self.convW_1  = nn.init.xavier_uniform_(nn.Parameter(w), gain=nn.init.calculate_gain('relu'))

      self.bias_1 = torch.nn.Parameter(torch.zeros(self.dim_capsule*self.n_channels), )

      w2 = torch.empty(self.dim_capsule*self.n_channels,self.dim_capsule,1, 1)
      self.CapsAct_W  = nn.init.xavier_uniform_(nn.Parameter(w2), gain=nn.init.calculate_gain('relu'))
      self.CapsAct_B = torch.nn.Parameter(torch.zeros(self.dim_capsule*self.n_channels))
      
      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

      #self.convW_1, self.bias_1, self.CapsAct_W, self.CapsAct_B  = self.convW_1.to(device), self.bias_1.to(device), self.CapsAct_W.to(device), self.CapsAct_B.to(device)

    def forward(self, inputs):
      conv1s = F.conv2d(inputs, self.convW_1,bias= self.bias_1, stride= self.stride, padding= 1) #padding = k-1/2
      conv1s = F.relu(conv1s)
      conv1s = torch.chunk(conv1s, self.n_channels, dim=1)
      CapsAct_ws = torch.chunk(self.CapsAct_W, self.n_channels, dim=0)
      CapsAct_bs = torch.chunk(self.CapsAct_B, self.n_channels, dim=-1)

      outputs = []
      for conv1, CapsAct_w, CapsAct_b in zip(conv1s, CapsAct_ws, CapsAct_bs):

        output = F.conv2d(conv1, CapsAct_w, bias=CapsAct_b, stride=1, padding= 0)
        output = output.unsqueeze(1)
        outputs.append(output)

      outputs = torch.cat(outputs, dim=1)
      return outputs

class Add_ConvCaps(nn.Module):
    def __init__(self,input_shape,layer_num,dim_capsule):
      super(Add_ConvCaps, self).__init__()
      self.layer_num = layer_num
      height = input_shape[3]
      width = input_shape[4]
      input_dim = input_shape[2]
      input_ch = input_shape[1]

      if layer_num ==2:
        self.ConvCaps1 = ConvCaps(input_shape = input_shape,n_channels=8, dim_capsule=dim_capsule, decrease_resolution = True)
        self.ConvCaps2 =   ConvCaps(input_shape = [None, input_ch, input_dim, int(height/2), int(width/2)],n_channels=8, dim_capsule=dim_capsule, decrease_resolution = False)
      
      elif layer_num ==3:
        self.ConvCaps1 = ConvCaps(input_shape = input_shape,n_channels=8, dim_capsule=dim_capsule, decrease_resolution = True)
        self.ConvCaps2 =   ConvCaps(input_shape = [None, input_ch, input_dim, int(height/2), int(width/2)],n_channels=8, dim_capsule=dim_capsule, decrease_resolution = False)
        self.ConvCaps3 =   ConvCaps(input_shape = [None, input_ch, input_dim, int(height/2), int(width/2)],n_channels=8, dim_capsule=dim_capsule, decrease_resolution = False)
  
    def forward(self,inputs):
      if self.layer_num==2:
        ConvCaps1 = F.tanh(self.ConvCaps1(inputs))
        output = F.tanh(self.ConvCaps2(ConvCaps1) + ConvCaps1)
        return output
        
      elif self.layer_num == 3:
        ConvCaps1 = F.tanh(self.ConvCaps1(inputs))
        ConvCaps2 = F.tanh(self.ConvCaps2(ConvCaps1) + ConvCaps1)
        output = F.tanh(self.ConvCaps3(ConvCaps2) + ConvCaps2)
        return output 

class ConvCaps(nn.Module):
    def __init__(self,input_shape, n_channels, dim_capsule, decrease_resolution=False):
      super(ConvCaps, self).__init__()
      self.n_channels = n_channels
      self.dim_capsule = dim_capsule
      if decrease_resolution == True:
        self.stride = 2
      else:
          self.stride = 1

      self.height = input_shape[3]
      self.width = input_shape[4]
      self.input_dim = input_shape[2]
      self.input_ch = input_shape[1]

      Att_W = torch.empty(self.input_ch*self.n_channels,self.input_ch,self.dim_capsule,1, 1)
      self.Att_W  = nn.init.xavier_uniform_(nn.Parameter(Att_W), gain=nn.init.calculate_gain('relu'))

      ConvTrans_W = torch.empty(self.input_ch*self.dim_capsule*self.n_channels,self.input_dim,3,3 )
      self.ConvTrans_W  = nn.init.xavier_uniform_(nn.Parameter(ConvTrans_W), gain=nn.init.calculate_gain('relu'))
      self.ConvTrans_B =  torch.nn.Parameter(torch.zeros(self.input_ch*self.dim_capsule*self.n_channels))

      CapsAct_W = torch.empty(self.dim_capsule*self.n_channels,self.dim_capsule,1, 1)
      self.CapsAct_W = nn.init.xavier_uniform_(nn.Parameter(CapsAct_W), gain=nn.init.calculate_gain('relu'))
      self.CapsAct_B =  torch.nn.Parameter(torch.zeros(self.dim_capsule*self.n_channels))  

      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, inputs):

      inputs = F.dropout( inputs, p=0.5)
      input_caps = torch.chunk(inputs, self.input_ch, dim=1)
      ConvTrans_ws = torch.chunk(self.ConvTrans_W, self.input_ch, dim=0)
      ConvTrans_bs = torch.chunk(self.ConvTrans_B, self.input_ch, dim=-1)

      # Convolutional Transform by 3x3 conv
      conv1s = [F.conv2d(torch.squeeze(input_cap, dim=1), ConvTrans_w,bias=ConvTrans_b, stride=self.stride, padding=1)
                  for input_cap, ConvTrans_w,ConvTrans_b in zip(input_caps, ConvTrans_ws,ConvTrans_bs)]
      conv1s = [torch.reshape( conv1, [-1, 1,self.n_channels, self.dim_capsule,int(self.height/self.stride), int(self.width/self.stride)])
                  for conv1 in conv1s]
      conv1s = torch.cat(conv1s, dim=1)
      conv1s = torch.transpose(conv1s, 2,1)
      
      Att_inputs = torch.chunk(conv1s, self.n_channels, dim=1)
      Att_ws = torch.chunk(self.Att_W, self.n_channels, dim=0)
      CapsAct_ws = torch.chunk(self.CapsAct_W, self.n_channels, dim=0)
      CapsAct_bs = torch.chunk(self.CapsAct_B, self.n_channels, dim=0)

      outputs = []
      for Att_input,Att_w,CapsAct_w,CapsAct_b in zip(Att_inputs,Att_ws, CapsAct_ws,CapsAct_bs):
        x = torch.squeeze(Att_input, dim=1) #x.shape = (batch_sz,input_ch,dim_cap, height, width)
        attentions = F.conv3d(x,Att_w)  # attentions shape =(batch_sz,input_ch,1, height, width)
        attentions = F.softmax(attentions,dim =1)
        final_attentions = torch.mul(x,attentions)
        final_attentions = torch.sum(final_attentions,dim = 1) #final_attentions.shape = (batch_sz,dim_cap,height, width)
        conv3 = F.conv2d(final_attentions,CapsAct_w,bias=CapsAct_b, padding= 0)
        conv3 = conv3.unsqueeze(1)
        outputs.append(conv3)
      outputs = torch.cat(outputs, dim=1)
      # print("ConvCapsOutput",outputs.shape)
      return outputs

class FullyConvCaps(nn.Module):
    def __init__(self,input_shape, n_channels, dim_capsule):
      super(FullyConvCaps, self).__init__()
      self.n_channels = n_channels
      self.dim_capsule = dim_capsule

      self.height = input_shape[3]
      self.width = input_shape[4]
      self.input_dim = input_shape[2]
      self.input_ch = input_shape[1]

      Att_W = torch.empty(self.input_ch*self.n_channels,self.input_ch,self.dim_capsule,1, 1)
      self.Att_W  = nn.init.xavier_uniform_(nn.Parameter(Att_W), gain=nn.init.calculate_gain('relu'))

      ConvTrans_W = torch.empty(self.input_ch*self.dim_capsule*self.n_channels,self.input_dim,self.height,self.width )
      self.ConvTrans_W  = nn.init.xavier_uniform_(nn.Parameter(ConvTrans_W), gain=nn.init.calculate_gain('relu'))
      self.ConvTrans_B =  torch.nn.Parameter(torch.zeros(self.input_ch*self.dim_capsule*self.n_channels))

      CapsAct_W = torch.empty(self.dim_capsule*self.n_channels,self.dim_capsule,1, 1)
      self.CapsAct_W = nn.init.xavier_uniform_(nn.Parameter(CapsAct_W), gain=nn.init.calculate_gain('relu'))
      self.CapsAct_B =  torch.nn.Parameter(torch.zeros(self.dim_capsule*self.n_channels))  

      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      #self.Att_W, self.ConvTrans_W, self.ConvTrans_B, self.CapsAct_W, self.CapsAct_B = self.Att_W.to(device), self.ConvTrans_W.to(device), self.ConvTrans_B.to(device), self.CapsAct_W.to(device), self.CapsAct_B.to(device) 
    def forward(self, inputs):

      inputs = F.dropout( inputs, p=0.5)
      input_caps = torch.chunk(inputs, self.input_ch, dim=1)
      ConvTrans_ws = torch.chunk(self.ConvTrans_W, self.input_ch, dim=0)
      ConvTrans_bs = torch.chunk(self.ConvTrans_B, self.input_ch, dim=-1)

      # Convolutional Transform by 3x3 conv
      conv1s = [F.conv2d(torch.squeeze(input_cap, dim=1), ConvTrans_w,bias=ConvTrans_b, stride=1)
                  for input_cap, ConvTrans_w,ConvTrans_b in zip(input_caps, ConvTrans_ws,ConvTrans_bs)]
      conv1s = [torch.reshape( conv1, [-1, 1,self.n_channels, self.dim_capsule,1, 1])
                  for conv1 in conv1s]
      conv1s = torch.cat(conv1s, dim=1)
      conv1s = torch.transpose(conv1s, 2,1)
      
      Att_inputs = torch.chunk(conv1s, self.n_channels, dim=1)
      Att_ws = torch.chunk(self.Att_W, self.n_channels, dim=0)
      CapsAct_ws = torch.chunk(self.CapsAct_W, self.n_channels, dim=0)
      CapsAct_bs = torch.chunk(self.CapsAct_B, self.n_channels, dim=0)

      outputs = []
      for Att_input,Att_w,CapsAct_w,CapsAct_b in zip(Att_inputs,Att_ws, CapsAct_ws,CapsAct_bs):
        x = torch.squeeze(Att_input, dim=1) #x.shape = (batch_sz,input_ch,dim_cap, height, width)
        attentions = F.conv3d(x,Att_w)  # attentions shape =(batch_sz,input_ch,1, height, width)
        attentions = F.softmax(attentions,dim =1)
        final_attentions = torch.mul(x,attentions)
        final_attentions = torch.sum(final_attentions,dim = 1) #final_attentions.shape = (batch_sz,dim_cap,height, width)
        conv3 = F.conv2d(final_attentions,CapsAct_w,bias=CapsAct_b)
        conv3 = conv3.unsqueeze(1)
        outputs.append(conv3)
      outputs = torch.cat(outputs, dim=1)
      outputs = torch.reshape(outputs, [-1, self.dim_capsule, self.n_channels])
      outputs = torch.transpose(outputs, 2, 1)
      # print(outputs.shape)
      return outputs