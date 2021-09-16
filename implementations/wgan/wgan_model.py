import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


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
    def __init__(self,batch_size, in_channels=1, dim=6, n_downsample=2):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        # Initial convolution block
        layers = [
            #nn.ReflectionPad3d(3),
            nn.Conv3d(in_channels, 8, 7),
            nn.InstanceNorm3d(8),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Downsampling

        layers += [
                nn.Conv3d(8, 16, 4, stride=3, padding=1),
                nn.InstanceNorm3d(16),
                nn.ReLU(inplace=True), nn.Conv3d(16, 32, 3, stride=3, padding=1),
                nn.InstanceNorm3d(32),
                nn.ReLU(inplace=True), nn.Conv3d(32, 64, 3, stride=3, padding=1),
                nn.InstanceNorm3d(64),
                nn.ReLU(inplace=True),nn.Conv3d(64, 128, 3, stride=3, padding=1),
                nn.InstanceNorm3d(128),
                nn.ReLU(inplace=True),nn.Conv3d(128, 256, 3, stride=3, padding=1),
                nn.InstanceNorm3d(256),
                nn.ReLU(inplace=True),nn.Conv3d(256, 512, 3, stride=3, padding=1),
                nn.InstanceNorm3d(512),
                nn.ReLU(inplace=True),nn.Conv3d(512, 1024, 3, stride=3, padding=1),
                nn.InstanceNorm3d(1024),
                nn.ReLU(inplace=True),
            ]


        self.model_blocks = nn.Sequential(*layers)


    def forward(self, x):
        z = self.model_blocks(x)
        return z.view((self.batch_size,z.numel() //self.batch_size ))


class Generator(nn.Module):
    def __init__(self, latent_dim,img_shape,batch_size):
        super(Generator, self).__init__()
        self.img_shape=img_shape
        self.batch_size=batch_size
        self.latent_dim =latent_dim
        self.init_size = img_shape[2] // 8
        self.l1 = nn.Sequential(nn.Linear(self.latent_dim, 128 * self.init_size ** 3))
        self.conv_blocks = nn.Sequential(
            nn.Conv3d(128, 64, 4, stride=1, padding=2),
            nn.BatchNorm3d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv3d(64, 32, 4, stride=1, padding=1),
            nn.BatchNorm3d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm3d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv3d(32, 16, 3, stride=1, padding=1),
            nn.BatchNorm3d(16, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv3d(16, 16, 3, stride=1),
            nn.BatchNorm3d(16, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(16, 3, kernel_size=3),
            nn.BatchNorm3d(3, 0.8),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0],128, self.init_size, self.init_size,self.init_size)
        img = self.conv_blocks(out)
        #img = img.view(self.batch_size,1,self.img_shape[2],self.img_shape[2],self.img_shape[2]).type(torch.float32)
        #img = img.permute(0, 2, 3, 4, 1).contiguous()

        img = img.view(img.numel() // 3 , 3)
        img = F.softmax(img)
        return img


class Discriminator(nn.Module):
    def __init__(self,img_shape,batch_size):
        super(Discriminator, self).__init__()
        self.img_shape= img_shape
        self.batch_size=batch_size
        self.model = nn.Sequential(
            nn.Linear(self.img_shape[2] ** 3, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(self.batch_size, -1)
        validity = self.model(img_flat)
        return validity


class WGAN(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, latent_dim, img_shape, batch_size):
        super(WGAN, self).__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.batch_size = batch_size
        self.Encoder = Encoder(self.batch_size)
        self.Generator = Generator(self.latent_dim,self.img_shape,self.batch_size)
        self.Discriminator = Discriminator(self.img_shape,self.batch_size)
        self.Generator.apply(weights_init_normal)
        self.Discriminator.apply(weights_init_normal)
        self.Encoder.apply(weights_init_normal)

    def forward(self, x):
        real_imgs = Variable(x.type(Tensor))
        encoder_imgs = self.Encoder(real_imgs)
        gen_img = self.Generator(encoder_imgs)
        gen_pred_flat = gen_img.data.max(1)[1]
        gen_pred= gen_pred_flat.view(self.batch_size,1,self.img_shape[1],self.img_shape[1],self.img_shape[1]).type(torch.float32)
        d_real = self.Discriminator(real_imgs)
        d_fake = self.Discriminator(gen_pred)
        return gen_img, gen_pred, d_real, d_fake
