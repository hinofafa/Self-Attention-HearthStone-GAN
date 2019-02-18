import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from spectral import SpectralNorm
import numpy as np

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)

        out = self.gamma*out + x
        return out,attention

class Generator(nn.Module):
    """Generator."""

    def __init__(self, batch_size, image_size=64, z_dim=100, conv_dim=64):
        super(Generator, self).__init__()
        self.imsize = image_size
        self.layers = []
        first_layer = []
        loop_layers = []
        attn_layers = []
        final_layer = []
        attn_feat=[16,32]

        n_layers = int(np.log2(self.imsize)) - 2
        mult = 8 #2 ** repeat_num  # 8
        assert mult * conv_dim > 3 * (2 ** n_layers), 'Need to add higher conv_dim, too many layers'

        curr_dim = conv_dim * mult

        # Initialize the first layer because it is different than the others.
        first_layer.append(SpectralNorm(nn.ConvTranspose2d(3, conv_dim, 4, 2, 1)))
        self.layers.append(nn.BatchNorm2d(curr_dim))
        first_layer.append(nn.ReLU())


        for n in range(n_layers - 1):
            loop_layers.append([])
            loop_layers[-1].append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
            loop_layers[-1].append(nn.BatchNorm2d(int(curr_dim / 2)))
            loop_layers[-1].append(nn.ReLU())
            if 2**(n+2) in attn_feat:
                attn_layers.append([])
                attn_layers[-1].append(Self_Attn(int(curr_dim / 2), 'relu'))
            curr_dim = int(curr_dim / 2)

        # append a final layer to change to 3 channels and add Tanh activation
        final_layer.append(nn.ConvTranspose2d(curr_dim, 3, 4, 2, 1))
        final_layer.append(nn.Tanh())

        self.layers.append(nn.Sequential(*first_layer))
        for n in range(n_layers - 1):
            self.layers.append(nn.Sequential(*loop_layers[n]))
            if n == 1:
                self.layers.append(attn_layers[0])
            if n == 2:
                self.layers.append(attn_layers[1])
        self.layers.append(nn.Sequential(*final_layer))

    def forward(self, z):
        #TODO add dynamic layers to the class for inspection. if this is done we can output p1 and p2, right now they
        # are a placeholder so training loop can be the same.
        z = z.view(z.size(0), z.size(1), 1, 1)
        out = torch.tensor([])
        p1 = 0
        p2 = 0
        for n in range(len(self.layers)):
            if n == 0:
                out = self.layers[n](z)
            else:
                out == self.layers[n](out)
                if n == 2:
                    out, p1 = self.layers[n](out)
                if n == 3:
                    out, p2 = self.layers[n](out)

        return out.squeeze(), p1, p2

class Discriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, batch_size=64, image_size=64, conv_dim=64, attn_feat=[8, 16]):
        super(Discriminator, self).__init__()
        self.imsize = image_size
        self.layers = nn.Parameter(torch.tensor([]))
        first_layer = []
        loop_layers = []
        attn_layers = []
        final_layer = []

        n_layers = int(np.log2(self.imsize)) - 2
        # Initialize the first layer because it is different than the others.
        first_layer.append(SpectralNorm(nn.Conv2d(3, conv_dim, 4, 2, 1)))
        first_layer.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        for n in range(n_layers - 1):
            loop_layers.append([])
            loop_layers[-1].append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
            loop_layers[-1].append(nn.LeakyReLU(0.1))
            curr_dim *= 2
            if 2**(n+2) in attn_feat:
                attn_layers.append([])
                attn_layers[-1].append(Self_Attn(curr_dim, 'relu'))

        final_layer.append(nn.Conv2d(curr_dim, 1, 4))

        self.layers.append(nn.Sequential(*first_layer))
        for n in range(n_layers - 1):
            self.layers.append(nn.Sequential(*loop_layers[n]))
            if n == 1:
                self.layers.append(attn_layers[0])
            if n == 2:
                self.layers.append(attn_layers[1])
        self.layers.append(nn.Parameter(*final_layer))

    def forward(self, x):
        out = torch.tensor([])
        p1 = 0
        p2 = 0
        for n in range(len(self.layers)):
            if n == 0:
                out = self.layers[n](x)
            else:
                out == self.layers[n](out)
                if n == 2:
                    out, p1 = self.layers[n](out)
                if n == 3:
                    out, p2 = self.layers[n](out)

        return out.squeeze(), p1, p2
