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

'''Resume est le chemin vers le modèle que l'ont veut utiliser'''

parser.add_argument("--resume", default=None)
parser.add_argument("--save", default=None)

''' 4 différents modes de fontionnement du scripts'''

'''Ligne : exploration de l'espace latent en ligne droite entre deux sujets'''

parser.add_argument("--ligne", default=False)
parser.add_argument("--nb_samples", default=10,type=int)

''' Feature rank : En argument le nombre de dimentions latentes, permet d'obtenir
un dictionnaire avec un score pour chaque dimension'''


parser.add_argument("--feature_rank", type=int, default=None)

''' sum_dim:  crée en .npy le cerveau moyen des cerveaux du batch_test encodés'''

parser.add_argument("--sum_dim", default=False)

''' expl_dim : explore une dimension de l'espace latent entre sa valeur maximale et minimale'''

parser.add_argument("--expl_dim",default=None, type=int)
parser.add_argument("--min_max", default=None,type=list)
opt = parser.parse_args()
print(opt)
text=''
if opt.min_max:
    for k in opt.min_max:
        text+=k

    text=text.split(',')
    extremums=[float(k) for k in text]
    print('range = ',extremums)


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = torch.device("cuda", index=0)
img_shape = (1, 80, 80,80)

if opt.save:
    if os.path.exists(opt.save):
        shutil.rmtree(opt.save)
    os.makedirs(opt.save, exist_ok=True)

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

_, skel_train, skel_val, skel_test = main_create('skeleton','L',batch_size = 1, nb = 1000,adn=False, directory_base='/neurospin/dico/data/deep_folding/data/crops/STS_branches/nearest/1mm/Lskeleton')

samples =[]

if opt.ligne:
    for i,sample in enumerate(skel_test):
        samples += Variable(sample[0].type(torch.Tensor)).to(device, dtype=torch.float32)
    save_image(samples[0][0,30,:,:], join(opt.save,"sample_img1.png"), nrow=5, normalize=True)
    save_image(samples[1][0,30,:,:], join(opt.save,"sample_img2.png"), nrow=5, normalize=True)
    enc_sample1=encoder(samples[0].unsqueeze(1)).to(device, dtype=torch.float32)
    enc_sample2=encoder(samples[1].unsqueeze(1)).to(device, dtype=torch.float32)
    diff= enc_sample2 - enc_sample1
    list_samples=[enc_sample1 + (k/opt.nb_samples)*diff for k in range(opt.nb_samples) ]

    for i,sample in enumerate(list_samples):
        gen_s=generator(sample)
        gen_max =gen_s.max(1)[1].to(torch.float32)
        save_image(gen_max[0,30,:,:], join(opt.save,"fake_img" + str(i)+".png"), nrow=5, normalize=True)

if opt.expl_dim != None:
    loss = nn.MSELoss()
    img=next(iter(skel_test))
    img = Variable(img[0].type(torch.Tensor)).to(device, dtype=torch.float32)
    img_encoded=encoder(img).to(device, dtype=torch.float32)
    list_samples= [torch.clone(img_encoded).detach() for k in range(opt.nb_samples) ]
    list_gen=[]
    espacement= (extremums[1] - extremums[0])/opt.nb_samples
    for k in range(opt.nb_samples):
        list_samples[k][0][opt.expl_dim] = k*espacement + extremums[0]
        gen_s=generator(list_samples[k])
        gen_max =gen_s.max(1)[1].to(torch.float32)
        list_gen+=[gen_max]
    losses=torch.Tensor([loss(img.squeeze(1),list_gen[k]) for k in range(opt.nb_samples)])
    print(torch.min(losses),torch.max(losses))
    for i,gen_max in enumerate(list_gen):
        save_image(gen_max[0,30,:,:], join(opt.save,"fake_img" + str(i)+".png"), nrow=5, normalize=True)

if opt.sum_dim:
    latents=torch.empty((len(skel_test),1728))
    for i,batch_test in enumerate(skel_test):
        real_imgs = Variable(batch_test[0].type(torch.Tensor)).to(device, dtype=torch.float32)

        z = encoder(real_imgs).detach().to(device, dtype=torch.float32)
        latents[i]=z
    sample_sum=torch.sum(latents,0)/len(skel_test)
    sample_sum=sample_sum.unsqueeze(0).to(device, dtype=torch.float32)
    gen_s=generator(sample_sum).to(device, dtype=torch.float32)
    gen_max =gen_s.max(1)[1].to(torch.float32).cpu().detach().numpy()
    print(gen_max.shape)
    #save_image(gen_max[0,30,:,:], "/neurospin/dico/adneves/wgan_gp/%s/%s.png" % (opt.save,"vect_moyen"), nrow=5, normalize=True)
    np.save(join(opt.save,"vectmoyen.npy"), gen_max)

if opt.feature_rank:
    W = torch.Tensor(3).to(device, dtype=torch.float32)
    W[0],W[1],W[2] = 1,1,10
    criterion_pixel =torch.nn.CrossEntropyLoss(weight = W)
    features_scores=dict()
    for dim in range(opt.feature_rank):
        print('test de la dimension %d' %(dim))
        score_dim = 0
        for batch_test in skel_test:
            real_imgs = Variable(batch_test[0].type(torch.Tensor)).to(device, dtype=torch.float32)
            z = encoder(real_imgs).detach().to(device, dtype=torch.float32)
            encoded_imgs= [torch.clone(z).detach() for k in range(3)]
            encoded_imgs[0][0][dim] -= 0.5
            encoded_imgs[2][0][dim] += 0.5
            scores = [criterion_pixel(generator(encoded), real_imgs.squeeze(1).long()).item() for encoded in encoded_imgs]
            score_dim += abs(scores[2] - scores[1]) + abs(scores[1] - scores[0])

        features_scores[('feature n°'+ str(dim)+' score' )] = [(score_dim*1000) / len(skel_test)]
        print(features_scores)


        #list_samples[k][0][opt.expl_dim] += k*espacement
