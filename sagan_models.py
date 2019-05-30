import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from spectral import SpectralNorm
import numpy as np

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            # TODO: InstanceNorm2d -> SpectralNorm
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)

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

    def __init__(self, batch_size, image_size=64, c_dim=5, conv_dim=64):
        super(Generator, self).__init__()
        self.imsize = image_size
        layerD1 = []
        layerD2 = []
        layerD3 = []
        layerBN = []
        layerU1 = []
        layerU2 = []
        layerU3 = []
        last = []
        
        ## Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,...)
        layerD1.append(SpectralNorm(nn.Conv2d(3+c_dim, conv_dim, 4, 2, 1)))
        layerD1.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        
        layerD2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layerD2.append(nn.ReLU(inplace=True))
        curr_dim = curr_dim * 2

        layerD3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layerD3.append(nn.ReLU(inplace=True))
        curr_dim = curr_dim * 2

        #if self.imsize == 64:
        #    layerD4 = []
        #    layerD4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        #    layerD4.append(nn.ReLU(inplace=True))
        #    self.lD4 = nn.Sequential(*layerD4)
        #    curr_dim = curr_dim * 2
        
        self.lD1 = nn.Sequential(*layerD1)
        self.lD2 = nn.Sequential(*layerD2)
        self.lD3 = nn.Sequential(*layerD3)
        #self.attnD1 = Self_Attn( 256, 'relu')
        #self.attnD2 = Self_Attn( 512, 'relu')

        # TODO calculate dim after attention@@

        # Bottleneck layers.
        repeat_num = 1 #int(np.log2(self.imsize)) - 3
        for i in range(repeat_num):  # 3 for imsize=image_size=64
            layerBN.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        self.BN = nn.Sequential(*layerBN)

        # Up-sampling layers.
        #mult = 4#2 ** (int(np.log2(self.imsize)) - 3) # 8. TODO Why does it work la QwQ
        ## ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1...)
        layerU1.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, curr_dim//2, 4, 2, 1))) # conv_dim*mult
        layerU1.append(nn.BatchNorm2d(curr_dim//2)) # conv_dim*mult
        layerU1.append(nn.ReLU())
        curr_dim = curr_dim//2 # conv_dim*mult

        layerU2.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, curr_dim//2, 4, 2, 1)))
        layerU2.append(nn.BatchNorm2d(curr_dim//2))
        layerU2.append(nn.ReLU())
        curr_dim = curr_dim//2

        #layerU3.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, curr_dim//2, 4, 2, 1)))
        #layerU3.append(nn.BatchNorm2d(curr_dim//2))
        #layerU3.append(nn.ReLU())
        #curr_dim = curr_dim//2

        #if self.imsize == 64:
        #    layerU4 = []
        #    layerU4.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, curr_dim//2, 4, 2, 1)))
        #    layerU4.append(nn.BatchNorm2d(curr_dim//2))
        #    layerU4.append(nn.ReLU())
        #    self.lU4 = nn.Sequential(*layerU4)
        #    curr_dim = curr_dim//2

        self.lU1 = nn.Sequential(*layerU1)
        self.lU2 = nn.Sequential(*layerU2)
        #self.lU3 = nn.Sequential(*layerU3)
        self.attnU1 = Self_Attn( 64, 'relu') # 128
        #self.attnU2 = Self_Attn( 64,  'relu')

        last.append(nn.ConvTranspose2d(curr_dim, 3, 4, 2, 1))
        last.append(nn.Tanh())
        self.last = nn.Sequential(*last)


    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)

        out=self.lD1(x)
        out=self.lD2(out)
        out=self.lD3(out)
        #out,pD1 = self.attnD1(out)
        #out=self.lD4(out)
        #out,pD2 = self.attnD2(out)

        out=self.BN(out)
        
        out=self.lU1(out)
        out=self.lU2(out)
        #out=self.lU3(out)
        out,pU1 = self.attnU1(out)
        #out=self.lU4(out)
        #out,pU2 = self.attnU2(out)
        
        out=self.last(out)

        return out, 'pD1', 'pD2', pU1, 'pU2'


class Discriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, batch_size=64, image_size=64, c_dim=5, conv_dim=64):
        super(Discriminator, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        conv1 = []
        conv2 = []

        layer1.append(SpectralNorm(nn.Conv2d(3, conv_dim, 4, 2, 1)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        #if self.imsize == 64:   # TODO why 64????
        #    layer4 = []
        #    layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        #    layer4.append(nn.LeakyReLU(0.1))
        #    self.l4 = nn.Sequential(*layer4)
        #    curr_dim = curr_dim*2
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        #repeat_num = int(np.log2(self.imsize)) - 3 #TODO
        #kernel_size = int(image_size / np.power(2, repeat_num))
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=8, stride=1, bias=False) #TODO kernal size was 3, padding=1
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=8, bias=False) #TODO kernal size was kernel_size

        self.attn1 = Self_Attn(256, 'relu') #TODO 256
        #self.attn2 = Self_Attn(512, 'relu')

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out,p1 = self.attn1(out)
        #out=self.l4(out)    # TODO what if we don't have l4 @@ Change to code to allow 128 la
        #out,p2 = self.attn2(out)
        out_src = self.conv1(out)
        out_cls = self.conv2(out)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1)), p1, 'p2'
