import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import shutil
from create_sets import *
from display_loss import *
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = torch.device("cuda", index=0)
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--save")
parser.add_argument("--img_size", type=int, default=80, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=20, help="interval betwen image samples")
parser.add_argument("--resume", default=None)
opt = parser.parse_args()
print(opt)


if os.path.exists(join('/neurospin/dico/adneves/wgan_gp/',opt.save)):
    shutil.rmtree(join('/neurospin/dico/adneves/wgan_gp/',opt.save))
os.makedirs(join('/neurospin/dico/adneves/wgan_gp/',opt.save), exist_ok=True)
try:
    os.mkdir(join('/neurospin/dico/adneves/wgan_gp/',opt.save, 'images'))
except:
    print ("dossier image déjà crée")

img_shape = (opt.channels, opt.img_size, opt.img_size,opt.img_size)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
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
        self.model2=nn.Sequential(nn.Conv3d(1, 3, kernel_size=1, stride=1))
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0],1, img_shape[2],img_shape[2],img_shape[2])
        img=self.model2(img)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(512000, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1).to(torch.float32)
        validity = self.model(img_flat)
        return validity

W = torch.Tensor(3)
W[0],W[1],W[2] = 1,1,2

criterion_pixel =torch.nn.CrossEntropyLoss(weight = W)
# Loss weight for gradient penalty
lambda_gp = 10

# Initialize generator and discriminator
generator = Generator().to(device, dtype=torch.float32)
discriminator = Discriminator().to(device, dtype=torch.float32)


optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))



def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1,1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# ----------
#  Training
# ----------
n_epoch=0
if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> chargement checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        generator.load_state_dict(checkpoint['state_dict_gen'])
        discriminator.load_state_dict(checkpoint['state_dict_disc'])
        n_epoch = checkpoint['epoch']
        print("=> chargé à l'époque n° (epoch {})"
              .format( checkpoint['epoch']))
    else:
        print("=> pas de checkpoint trouvé '{}'".format(opt.resume))


loss_gen,loss_disc=[],[]
i = 0
batches_done = 0
_, skel_train, skel_val, skel_test = main_create('skeleton','L',batch_size = opt.batch_size, nb = 1000,adn=False, directory_base='/neurospin/dico/deep_folding_data/data/crops/STS_branches/nearest/original/Lskeleton')
for epoch in range(opt.n_epochs):
    for batch_skel in skel_train:

        # Configure input
        torch.cuda.empty_cache()
        real_imgs = Variable(batch_skel[0].type(torch.Tensor)).to(device, dtype=torch.float32)
        img_norm=torch.clone(real_imgs).to(device, dtype=torch.float32)
        img_norm[img_norm==11]=1
        img_norm[img_norm==60]=2
        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim)))).to(device, dtype=torch.float32)

        # Generate a batch of images
        fake_imgs = generator(z)
        fake_maxed = fake_imgs.data.max(1)[1].unsqueeze(1).to(torch.float32)
        loss_CE=criterion_pixel(fake_maxed, img_norm.squeeze(1).long())

        # Real images
        real_validity = discriminator(img_norm)
        # Fake images
        fake_validity = discriminator(fake_maxed)
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, img_norm.data, fake_maxed.data)
        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()

        # Train the generator every n_critic steps
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            #fake_imgs = generator(z)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            #fake_validity = discriminator(fake_imgs)
            g_loss = -torch.mean(fake_validity) + loss_CE

            g_loss.backward()
            optimizer_G.step()
            loss_disc += [d_loss.item()]
            loss_gen += [g_loss.item()]

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch+n_epoch, n_epoch+opt.n_epochs, i, len(skel_train), d_loss.item(), g_loss.item())
            )

            if batches_done % opt.sample_interval == 0:

                real_imgs[real_imgs==1]=11
                real_imgs[real_imgs==2]=60
                save_image(real_imgs[0,0,30,:,:], "/neurospin/dico/adneves/wgan_gp/%s/%s/%s.png" % (opt.save,"images","target" + str(epoch) + '_' + str(batches_done)), nrow=5, normalize=True)
                save_image(fake_maxed[0,0,30,:,:], "/neurospin/dico/adneves/wgan_gp/%s/%s/%s.png" % (opt.save,"images","fake_img" + str(epoch) + '_' + str(batches_done)), nrow=5, normalize=True)

            batches_done += opt.n_critic
        i += 1
print('saving ... ')
state={'epoch': n_epoch + opt.n_epochs, 'state_dict_gen': generator.state_dict(),'state_dict_disc': discriminator.state_dict()}
name=join('/neurospin/dico/adneves/wgan_gp/', 'epoch_'+str(n_epoch + opt.n_epochs)+ '_checkpoint.pth.tar')
torch.save(state, name)
print('model saved to ' + name)
display_loss(loss_disc, loss_gen)
