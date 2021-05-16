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
    def __init__(self,batch_size, in_channels=1, dim=16, n_downsample=2):
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
        z = self.model_blocks(x)
        #z = self.reparameterization(x)
        return z.view((self.batch_size,z.numel() //self.batch_size ))


class Generator(nn.Module):
    def __init__(self, latent_dim,img_shape):
        super(Generator, self).__init__()
        self.latent_dim =latent_dim
        self.init_size = img_shape[1] // 8
        self.l1 = nn.Sequential(nn.Linear(self.latent_dim, 128 * self.init_size ** 3))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm3d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv3d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm3d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv3d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm3d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv3d(64, 3, kernel_size=1),
            nn.BatchNorm3d(3, 0.8),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size,self.init_size)
        img = self.conv_blocks(out)
        img = img.permute(0, 2, 3, 4, 1).contiguous()
                # flatten
        img = img.view(img.numel() // 3 , 3)
        img = F.softmax(img)
        return img


class Discriminator(nn.Module):
    def __init__(self, batch_size,img_shape):
        super(Discriminator, self).__init__()
        self.img_shape = img_shape
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv3d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout3d(0.25)]
            if bn:
                block.append(nn.BatchNorm3d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(1, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_shape[1] // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 3, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


class dcGAN(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, latent_dim, img_shape, batch_size):
        super(dcGAN, self).__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.batch_size = batch_size
        self.Encoder = Encoder(self.batch_size)
        self.Generator = Generator(self.latent_dim,self.img_shape)
        self.Discriminator = Discriminator(self.batch_size,self.img_shape)
        self.Generator.apply(weights_init_normal)
        self.Discriminator.apply(weights_init_normal)
        self.Encoder.apply(weights_init_normal)

    def forward(self, x):
        real_imgs = Variable(x.type(Tensor))
        encoder_imgs = self.Encoder(real_imgs)
        gen_img = self.Generator(encoder_imgs)
        gen_pred_flat = gen_img.data.max(1)[1]
        gen_pred= gen_pred_flat.view(self.batch_size,1,self.img_shape[1],self.img_shape[1],self.img_shape[1]).type(torch.float32)
        return gen_img,gen_pred
