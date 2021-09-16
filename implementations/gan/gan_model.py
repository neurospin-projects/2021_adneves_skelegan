import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


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
    def __init__(self,batch_size, in_channels=1, dim=8, n_downsample=2, shared_block=None):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
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
                nn.Conv3d(dim, dim * 2, 4, stride=7, padding=1),
                nn.InstanceNorm3d(dim * 2),
                nn.ReLU(inplace=True),
            ]
            dim *= 2

        # Residual blocks
        for _ in range(3):
            layers += [ResidualBlock(dim)]

        self.model_blocks = nn.Sequential(*layers)

    def reparameterization(self, mu):
        Tensor = torch.cuda.FloatTensor if mu.is_cuda else torch.FloatTensor
        z = Variable(Tensor(np.random.normal(0, 1, mu.shape)))
        return z + mu

    def forward(self, x):
        x = self.model_blocks(x)
        z = self.reparameterization(x)
        return z.view((self.batch_size,z.numel() //self.batch_size ))


class Generator(nn.Module):
    def __init__(self, latent_dim,img_shape):
        super(Generator, self).__init__()
        self.latent_dim =latent_dim
        self.img_shape = img_shape
        def block(in_feat, out_feat, normalize=False):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, batch_size,img_shape):
        super(Discriminator, self).__init__()
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.model = nn.Sequential(
            nn.Linear(img_shape[2] ** 3, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(self.batch_size, self.img_shape[2] ** 3)
        #print('img flat : ', img_flat.size)
        validity = self.model(img_flat)

        return validity

class GAN(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, latent_dim, img_shape, batch_size):
        super(GAN, self).__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.batch_size = batch_size
        self.Encoder = Encoder(self.batch_size)
        self.Generator = Generator(self.latent_dim,self.img_shape)
        self.Discriminator = Discriminator(self.batch_size,self.img_shape)

    def forward(self, x):
        real_imgs = Variable(x.type(Tensor))
        encoder_imgs = self.Encoder(real_imgs)
        gen_imgs = self.Generator(encoder_imgs)
        d_real = self.Discriminator(real_imgs)
        d_fake = self.Discriminator(gen_imgs.detach())
        return gen_imgs, d_real, d_fake
