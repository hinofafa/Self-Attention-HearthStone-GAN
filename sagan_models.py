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
        # print('proj_query size: ', proj_query.size())
        # print('proj_key size: ', proj_key.size())
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        # print('energy size: ', energy.size())
        attention = self.softmax(energy) # B X (N) X (N)
        # print('attention size: ', attention.size())
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
        layer1 = []
        layer2 = []
        layer3 = []
        layer4 = []
        layer5 = []
        layer5 = []
        # layer6 = []
        last = []

        repeat_num = int(np.log2(self.imsize)) - 3
        mult = 2 ** repeat_num # 8
        layer1.append(SpectralNorm(nn.ConvTranspose2d(z_dim, int(conv_dim * mult), 8)))
        layer1.append(nn.BatchNorm2d(int(conv_dim * mult)))
        layer1.append(nn.ReLU())

        curr_dim = int(conv_dim * mult)

        layer2.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer2.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer2.append(nn.ReLU())

        curr_dim = int(curr_dim / 2)

        layer3.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer3.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer3.append(nn.ReLU())

        curr_dim = int(curr_dim / 2)

        layer4.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer4.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer4.append(nn.ReLU())

        curr_dim = int(curr_dim / 2)

        layer5.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer5.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer5.append(nn.ReLU())

        curr_dim = int(curr_dim / 2)

        # layer6.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        # layer6.append(nn.BatchNorm2d(int(curr_dim / 2)))
        # layer6.append(nn.ReLU())
        #
        # curr_dim = int(curr_dim / 2)

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)
        self.l4 = nn.Sequential(*layer4)
        self.l5 = nn.Sequential(*layer5)
        # self.l6 = nn.Sequential(*layer6)

        last.append(nn.ConvTranspose2d(curr_dim, 3, 4, 2, 1))
        last.append(nn.Tanh())
        self.last = nn.Sequential(*last)

        # self.attn1 = Self_Attn( 16, 'relu')
        self.attn2 = Self_Attn( 32, 'relu')

    def forward(self, z):
        # print('*****Generator*****')
        z = z.view(z.size(0), z.size(1), 1, 1)
        # print('input size: ', z.size())
        out=self.l1(z)
        # print('gl1 size: ', out.size())
        out=self.l2(out)
        # print('gl2 size: ', out.size())
        out=self.l3(out)
        # print('gl3 size: ', out.size())
        out=self.l4(out)
        # print('l4 size: ', out.size())
        # out,p1 = self.attn1(out)
        # print('dattn1 size: ', out.size())
        out = self.l5(out)
        # print('gl4 size: ', out.size())
        out,p2 = self.attn2(out)
        # print('gattn2 size: ', p2.size())
        out=self.last(out)
        # print('glast size: ', out.size())
        # return out, p1, p2
        return out, p2

class Discriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, batch_size=64, image_size=64, conv_dim=64):
        super(Discriminator, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        layer4 = []
        layer5 = []
        last = []

        layer1.append(SpectralNorm(nn.Conv2d(3, conv_dim, 4, 2, 1)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim #128

        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2 #64

        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2 #32

        layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer4.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim*2 #16

        layer5.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer5.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim*2 #8

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)
        self.l4 = nn.Sequential(*layer4)
        self.l5 = nn.Sequential(*layer5)

        last.append(nn.Conv2d(curr_dim, 1, 8))
        self.last = nn.Sequential(*last)

        # self.attn1 = Self_Attn(512, 'relu')
        self.attn2 = Self_Attn(1024, 'relu')

    def forward(self, x):
        # print('*****Discriminator*****')
        # print('x size: ', x.size())
        out = self.l1(x)
        # print('dl1 size: ', out.size())
        out = self.l2(out)
        # print('dl2 size: ', out.size())
        out = self.l3(out)
        # print('dl3 size: ', out.size())
        out=self.l4(out)
        # print('l4 size: ', out.size())
        # out,p1 = self.attn1(out)
        # print('dattn1 size: ', out.size())
        out = self.l5(out)
        # print('dl4 size: ', out.size())
        out,p2 = self.attn2(out)
        # print('dattn2 size: ', p2.size())
        out=self.last(out)
        # print('dlast size: ', out.size())
        return out.squeeze()
