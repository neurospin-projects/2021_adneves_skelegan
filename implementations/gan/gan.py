import argparse
import os
import numpy as np
import math

#import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from create_sets import *
from display_loss import *
import shutil
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images", exist_ok=True)
device = torch.device("cuda", index=0)
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--generation", type=int, default = 0)
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=256, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=96, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument("--save")
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')


opt = parser.parse_args()
print(opt)

if os.path.exists(opt.save):
    shutil.rmtree(opt.save)
os.makedirs(opt.save, exist_ok=True)
try:
    os.mkdir(join(opt.save, 'images'))
except:
    print ("dossier image déjà crée")

img_shape = (opt.channels, opt.img_size, opt.img_size,opt.img_size)

cuda = True if torch.cuda.is_available() else False

def save_checkpoint(state, is_best, path, filename='checkpoint.pth.tar'):
    prefix_save = os.path.join('/home/ad265693/GAN/implementations/Vnet', path)
    name = prefix_save + '_' + filename
    print('saving ... ')
    torch.save(state, name)
    print('model saved to ' + name)
    if is_best:
        shutil.copyfile(name, prefix_save + '_model_best.pth.tar')

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
    def __init__(self, in_channels=1, dim=8, n_downsample=2, shared_block=None):
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
        return z.view((opt.batch_size,z.numel() //opt.batch_size ))


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=False):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(img_shape[2] ** 3, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(opt.batch_size, img_shape[2] ** 3)
        #print('img flat : ', img_flat.size)
        validity = self.model(img_flat)

        return validity


# Loss function
lambda_e = 100
criterion_pixel = torch.nn.L1Loss()
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
encoder = Encoder().to(device, dtype=torch.float32)
generator = Generator().to(device, dtype=torch.float32)
discriminator = Discriminator().to(device, dtype=torch.float32)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Optimizers
optimizer_E = torch.optim.Adam(encoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

loss_disc, loss_gen, loss_enc = [], [], []
for epoch in range(opt.n_epochs):
    i = 0
    _, skel_train, skel_val, skel_test = main_create('skeleton','L',batch_size = opt.batch_size, nb = 1000)
    for batch_skel in skel_train:

        target = Variable(batch_skel[0].type(torch.Tensor)).to(device, dtype=torch.float32)
        # Adversarial ground truths
        valid = Variable(Tensor(target.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(target.shape[0], 1).fill_(0.0), requires_grad=False)
        # Configure input
        real_imgs = Variable(target.type(Tensor))

        # ---------------------
        #  Train Encoder
        # ---------------------

        optimizer_E.zero_grad()

        encoder_imgs = encoder(real_imgs)
        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (target.shape[0], opt.latent_dim))))
        # Generate a batch of images
        gen_imgs = generator(encoder_imgs)

        # Loss measures generator's ability to fool the discriminator
        e_loss = lambda_e * criterion_pixel(real_imgs,gen_imgs)
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        e_loss.backward(retain_graph=True)
        g_loss.backward(retain_graph=True)
        optimizer_E.step()
        optimizer_G.step()


        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()
        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward(retain_graph=True)
        optimizer_D.step()
        loss_disc += [d_loss]
        loss_gen += [g_loss]
        loss_enc += [e_loss]
        i += 1

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [E loss: %f]"
            % (epoch, opt.n_epochs, i, len(skel_train), d_loss.item(), g_loss.item(), e_loss.item())
        )

        batches_done = epoch * len(skel_train) + i
        torch.cuda.empty_cache()
        if batches_done % opt.sample_interval == 0:
            save_image(target[0,0,50,:,:], "/home/ad265693/GAN/implementations/gan/%s/%s/%s.png" % (opt.save,"images","data" + str(epoch) + '_' + str(batches_done)), nrow=5, normalize=True)
            save_image(gen_imgs[0,0,50,:,:], "/home/ad265693/GAN/implementations/gan/%s/%s/%s.png" % (opt.save,"images","target" + str(epoch) + '_' + str(batches_done)), nrow=5, normalize=True)
display_loss(loss_disc, loss_gen,loss_enc)
save_checkpoint({'epoch': epoch,
                     'state_dict': model.state_dict(),
                     'best_prec1': best_prec1},
                     opt.save, "gan")



#### TEST
_, skel_train, skel_val, skel_test = main_create('skeleton','L',batch_size = opt.batch_size, nb = 1000)
loss_enc=0
for batch_skel in skel_test:
    target = Variable(batch_skel[0].type(torch.Tensor)).to(device, dtype=torch.float32)

    # Configure input
    real_imgs = Variable(target.type(Tensor))
    encoder_imgs = encoder(real_imgs)

    # Generate a batch of images
    gen_imgs = generator(encoder_imgs)

    e_loss = lambda_e * criterion_pixel(real_imgs,gen_imgs)

    loss_enc += e_loss
print('loss de reconstruction en test : ', loss_enc)

## génération de squelettes nouveaux
if opt.generation !=0:
    for new_im in range(opt.generation):
        z = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))
        gen_imgs = generator(z)
        save_image(gen_imgs[0,0,50,:,:], "/home/ad265693/GAN/implementations/gan/%s/%s/%s.png" % (opt.save,"images","new_im_" + str(new_im)), nrow=5, normalize=True)
