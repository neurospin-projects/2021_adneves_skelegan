# avec seaborn
import seaborn as sn
import argparse
import os
import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torchvision.utils import save_image

from torch.utils.data import DataLoader,Subset
from torch.autograd import Variable
from create_sets import *
import torch
import torch.nn as nn


'''Permet d'explorer 3 dimensions et resort ses histogrammes ainsi que ses valeurs maximales et minimales'''

parser = argparse.ArgumentParser()
'''Resume est le chemin vers le modèle que l'ont veut utiliser'''
parser.add_argument("--resume", default=None)
parser.add_argument("--dimensions", default=None, type=list)
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = torch.device("cuda", index=0)
img_shape = (1, 80, 80,80)
twodim=False
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

encoder= Encoder(1)
if os.path.isfile(opt.resume):
    print("=> chargement checkpoint '{}'".format(opt.resume))
    checkpoint = torch.load(opt.resume)
    encoder.load_state_dict(checkpoint['state_dict_enc'])
    n_epoch = checkpoint['epoch']
    print("=> chargé à l'époque n° (epoch {})"
              .format( checkpoint['epoch']))
else:
    print("=> pas de checkpoint trouvé '{}'".format(opt.resume))

_, skel_train, skel_val, skel_test = main_create('skeleton','L',batch_size = 1, nb = 1000,adn=False, directory_base='/neurospin/dico/data/deep_folding/data/crops/STS_branches/nearest/original/Lskeleton')
latents=np.empty((len(skel_test),1728))

for i,batch_test in enumerate(skel_test):
    real_imgs = Variable(batch_test[0].type(torch.Tensor))

    z = encoder(real_imgs).detach().squeeze(0).numpy()
    latents[i]=z


range_values_dim=[]
text=''
for k in opt.dimensions:
    text+=k

text=text.split(',')
dimensions=[int(k) for k in text]
print('dimensions = ',dimensions)
for latent_dim in range(1728):
    sn.kdeplot([float(latents[k][latent_dim]) for k in range(len(skel_test))])

# latents correspond à un array (n_samples x n_dim) (1000 x 100) dans mon cas# avec des histogrammes de matplotlib (moins lisibles si tu veux afficher pour plusieurs dim

g1= [k[dimensions[0]] for k in latents]
g2= [k[dimensions[1]] for k in latents]
g3= [k[dimensions[2]] for k in latents]
print('dim ' + str(dimensions[0]) + ' range : [',min(g1),',',max(g1),']')
print('dim ' + str(dimensions[1]) + ' range : [',min(g2),',',max(g2),']')
print('dim ' + str(dimensions[2]) + ' range : [',min(g3),',',max(g3),']')
fig = plt.figure()
g1=plt.hist(g1, bins=20,label='dim ' + str(dimensions[0]))
g2=plt.hist(g2, bins=20,label='dim ' + str(dimensions[1]))
g3=plt.hist(g3, bins=20,label='dim ' + str(dimensions[2]))
plt.gca().set(title='Frequency Histogram', ylabel='Frequency')
plt.gca().legend()
plt.show()


'''
for latent_dim in range(100):
    plt.hist([float(latents[k][0][latent_dim]) for k in range(1000)], bins=50)
    plt.gca().set(title='Frequency Histogram', ylabel='Frequency');
    '''
