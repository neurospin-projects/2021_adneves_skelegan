import argparse
import os
import numpy as np
import math
import sys

from torchvision.utils import save_image

from torch.utils.data import DataLoader,Subset
from torchvision import datasets
from torch.autograd import Variable
import shutil
import random
from create_sets import *
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
parser = argparse.ArgumentParser()
parser.add_argument("--resume", default=None)
parser.add_argument("--save")
parser.add_argument("--nb_samples", default=10,type=int)
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = torch.device("cuda", index=0)
img_shape = (1, 80, 80,80)

if os.path.exists(join('/neurospin/dico/adneves/wgan_gp/',opt.save)):
    shutil.rmtree(join('/neurospin/dico/adneves/wgan_gp/',opt.save))
os.makedirs(join('/neurospin/dico/adneves/wgan_gp/',opt.save), exist_ok=True)

class Encoder(nn.Module):
    def __init__(self,batch_size, in_channels=1, dim=8, n_downsample=3):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        # Initial convolution block
        layers = [
            #nn.ReflectionPad3d(3),
            nn.Conv3d(in_channels, dim, 3),
            nn.BatchNorm3d(dim),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Downsampling
        for _ in range(n_downsample):
            layers += [
                nn.Conv3d(dim, dim * 2, 4, stride=3, padding=1),
                nn.BatchNorm3d(dim * 2),
                nn.ReLU(inplace=True),
            ]
            dim *= 2


        self.model_blocks = nn.Sequential(*layers)


    def forward(self, x):
        z = self.model_blocks(x)
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
        img = self.l1(z)
        img = img.view(img.shape[0], 128, self.init_size, self.init_size,self.init_size)
        img = self.conv_blocks(img)
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

encoder= Encoder(1).to(device, dtype=torch.float32)
generator = Generator(1728,img_shape).to(device, dtype=torch.float32)
discriminator = Discriminator().to(device, dtype=torch.float32)

if os.path.isfile(opt.resume):
    print("=> chargement checkpoint '{}'".format(opt.resume))
    checkpoint = torch.load(opt.resume)
    generator.load_state_dict(checkpoint['state_dict_gen'])
    discriminator.load_state_dict(checkpoint['state_dict_disc'])
    encoder.load_state_dict(checkpoint['state_dict_enc'])
    n_epoch = checkpoint['epoch']
    print("=> chargé à l'époque n° (epoch {})"
              .format( checkpoint['epoch']))
else:
    print("=> pas de checkpoint trouvé '{}'".format(opt.resume))


print('début du test')
_, skel_train, skel_val, skel_test = main_create('skeleton','L',batch_size = 1, nb = 1000,adn=False, directory_base='/neurospin/dico/deep_folding_data/data/crops/STS_branches/nearest/original/Lskeleton')
idx_1=random.randint(0,110)
idx_2=random.randint(0,110)
samples =[]
for i,sample in enumerate(skel_test):
    samples += Variable(sample[0].type(torch.Tensor)).to(device, dtype=torch.float32)
save_image(samples[0][0,30,:,:], "/neurospin/dico/adneves/wgan_gp/%s/%s.png" % (opt.save,"sample_img1"), nrow=5, normalize=True)
save_image(samples[1][0,30,:,:], "/neurospin/dico/adneves/wgan_gp/%s/%s.png" % (opt.save,"sample_img2"), nrow=5, normalize=True)
enc_sample1=encoder(samples[0].unsqueeze(1)).to(device, dtype=torch.float32)
enc_sample2=encoder(samples[1].unsqueeze(1)).to(device, dtype=torch.float32)
diff= enc_sample2 - enc_sample1
list_samples=[enc_sample1 + (k/opt.nb_samples)*diff for k in range(opt.nb_samples) ]

for i,sample in enumerate(list_samples):
    gen_s=generator(sample)
    gen_max =gen_s.max(1)[1].to(torch.float32)
    save_image(gen_max[0,30,:,:], "/neurospin/dico/adneves/wgan_gp/%s/%s.png" % (opt.save,"fake_img" + str(i)), nrow=5, normalize=True)
