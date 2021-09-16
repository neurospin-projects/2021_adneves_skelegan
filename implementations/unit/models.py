import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm3d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


##############################
#           RESNET
##############################


class ResidualBlock(nn.Module):
    def __init__(self, features):
        super(ResidualBlock, self).__init__()

        conv_block1 = [
            #nn.ReflectionPad3d(1),
            nn.Conv3d(features, features, 3),
            nn.InstanceNorm3d(features),
            nn.ReLU(inplace=True),
            #nn.ReflectionPad3d(1),
        ]
        conv_block2 = [
            nn.Conv3d(features, features, 3),
            nn.InstanceNorm3d(features),
        ]

        self.conv_block1 = nn.Sequential(*conv_block1)
        self.conv_block2 = nn.Sequential(*conv_block2)

    def forward(self, x):
        x_c = F.pad(self.conv_block1(F.pad(x,(1,1,1,1,1,1))),(1,1,1,1,1,1))
        return x + self.conv_block2(x_c)


class Encoder(nn.Module):
    def __init__(self, in_channels=1, dim=64, n_downsample=2, shared_block=None):
        super(Encoder, self).__init__()

        # Initial convolution block
        layers = [
            #nn.ReflectionPad3d(3),
            nn.Conv3d(in_channels, dim, 7),
            nn.InstanceNorm3d(dim),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Downsampling
        for _ in range(n_downsample):
            layers += [
                nn.Conv3d(dim, dim * 2, 4, stride=2, padding=1),
                nn.InstanceNorm3d(dim * 2),
                nn.ReLU(inplace=True),
            ]
            dim *= 2

        # Residual blocks
        for _ in range(3):
            layers += [ResidualBlock(dim)]

        self.model_blocks = nn.Sequential(*layers)
        self.shared_block = shared_block

    def reparameterization(self, mu):
        Tensor = torch.cuda.FloatTensor if mu.is_cuda else torch.FloatTensor
        z = Variable(Tensor(np.random.normal(0, 1, mu.shape)))
        return z + mu

    def forward(self, x):
        x = self.model_blocks(x)
        mu = self.shared_block(x)
        z = self.reparameterization(mu)
        print('tespace l',z.shape)
        return mu, z


class Generator(nn.Module):
    def __init__(self, out_channels=3, dim=64, n_upsample=2, shared_block=None):
        super(Generator, self).__init__()

        self.shared_block = shared_block
        self.dim=dim

        layers = []
        dim = dim * 2 ** n_upsample
        # Residual blocks
        for _ in range(3):
            layers += [ResidualBlock(dim)]

        # Upsampling
        for _ in range(n_upsample):
            layers += [
                nn.ConvTranspose3d(dim, dim // 2, 4, stride=2, padding=0),
                nn.InstanceNorm3d(dim // 2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            dim = dim // 2

        # Output layer
        layers2 = [nn.Conv3d(dim, out_channels, 7), nn.Tanh()]
        #layers += [nn.ReflectionPad3d(3), nn.Conv3d(dim, out_channels, 7), nn.Tanh()]

        self.model_blocks1 = nn.Sequential(*layers)
        self.model_blocks2 = nn.Sequential(*layers2)

    def forward(self, x):
        #print("début : ", x.shape)
        x = self.shared_block(x)
        #print("après shared : ",x.shape)
        x = self.model_blocks1(x)
        #print("après block 1 : ",x.shape)
        x = F.pad(x,(6,5,6,5,5,5))
        #print("après pad : ",x.shape)
        x = self.model_blocks2(x)
        #print("après block2 : ",x.shape)
        return x


##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, input_shape,dim):
        super(Discriminator, self).__init__()
        channels, height, width, depth = input_shape
        # Calculate output of image discriminator (PatchGAN)
        #self.output_shape = (1, height // 2 ** 4, width // 2 ** 4, depth // 2 ** 4)
        self.output_shape = (1, 4, 5, 2)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv3d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm3d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, dim, normalize=False),
            *discriminator_block(dim, dim*2),
            *discriminator_block(dim*2, dim*4),
            *discriminator_block(dim*4, dim*8),
            nn.Conv3d(dim * 8, 1, 3, padding=1)
        )

    def forward(self, img):
        return self.model(img)
