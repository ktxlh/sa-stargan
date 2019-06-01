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
        
        layer1 = []
        # 3x64x64 -> 64x32x32
        layer1.append(SpectralNorm(nn.Conv2d(3+c_dim, conv_dim, 4, 2, 1)))
        layer1.append(nn.BatchNorm2d(conv_dim))
        layer1.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim

        # 64x32x32 -> 128x16x16
        layer1.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer1.append(nn.BatchNorm2d(curr_dim * 2))
        layer1.append(nn.ReLU(inplace=True))
        curr_dim = curr_dim * 2
        self.l1 = nn.Sequential(*layer1)
        
        # 128x16x16
        self.attn1 = Self_Attn( 128, 'relu') # 256

        # 128x16x16 -> 256x8x8
        layer2 = []
        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer2.append(nn.BatchNorm2d(curr_dim * 2))
        layer2.append(nn.ReLU(inplace=True))
        curr_dim = curr_dim * 2
        
        # Bottleneck layers. For testing whether sa-stargan outperforms stargan
        #repeat_num = 1 #int(np.log2(self.imsize)) - 3
        #for _ in range(1):  # 3 for imsize=image_size=64
        #    layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        # 256x8x8 -> 128x16x16
        layer2.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, curr_dim//2, 4, 2, 1)))
        layer2.append(nn.BatchNorm2d(curr_dim//2))
        layer2.append(nn.ReLU())
        curr_dim = curr_dim//2
        self.l2 = nn.Sequential(*layer2)
        
        # 128x16x16
        self.attn2 = Self_Attn( 128, 'relu')

        # 128x16x16 -> 64x32x32
        layer3 = []
        layer3.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, curr_dim//2, 4, 2, 1)))
        layer3.append(nn.BatchNorm2d(curr_dim//2))
        layer3.append(nn.ReLU())
        curr_dim = curr_dim//2

        layer3.append(nn.ConvTranspose2d(curr_dim, 3, 4, 2, 1))
        layer3.append(nn.Tanh())

        self.l3 = nn.Sequential(*layer3)


    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        out = self.l1(x)
        out, p1 = self.attn1(out)
        out = self.l2(out)
        out, p2 = self.attn2(out)
        out = self.l3(out)
        return out, 'pD1', 'pD2', 'pU1', 'pU2'


class Discriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, batch_size=64, image_size=64, c_dim=5, conv_dim=64):
        super(Discriminator, self).__init__()
        self.imsize = image_size
        
        layer1 = []
        # 3x64x64 -> 64x32x32
        layer1.append(SpectralNorm(nn.Conv2d(3, conv_dim, 4, 2, 1)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        # 64x32x32 -> 128x16x16
        layer1.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer1.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2
        self.l1 = nn.Sequential(*layer1)

        # 128x16x16
        self.attn1 = Self_Attn(128, 'relu')

        layer2 = []
        # 128x16x16 -> 256x8x8
        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2
        self.l2 = nn.Sequential(*layer2)

        # 256x8x8
        #self.attn2 = Self_Attn(256, 'relu')

        # 256x8x8 -> 512x4x4
        layer3 = []
        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2
        self.l3 = nn.Sequential(*layer3)

        #repeat_num = int(np.log2(self.imsize)) - 3 #TODO
        #kernel_size = int(image_size / np.power(2, repeat_num))
        self.conv_scr = nn.Conv2d(curr_dim, 1,     kernel_size=4, bias=False) #TODO kernal size was 3, padding=1
        self.conv_cls = nn.Conv2d(curr_dim, c_dim, kernel_size=4, bias=False) #TODO kernal size was kernel_size
        

    def forward(self, x):
        out = self.l1(x)
        out, p1 = self.attn1(out)
        out = self.l2(out)
        #out, p2 = self.attn2(out)
        out = self.l3(out)
        out_src = self.conv_scr(out)
        out_cls = self.conv_cls(out)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1)), 'p1', 'p2'
