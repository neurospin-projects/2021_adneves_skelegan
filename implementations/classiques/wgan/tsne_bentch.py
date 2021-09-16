from sklearn.manifold import TSNE
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

parser = argparse.ArgumentParser()
parser.add_argument("--resume", default=None)
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = torch.device("cuda", index=0)
img_shape = (1, 80, 80,80)

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

tsne = TSNE(n_components=2, perplexity=40, n_iter=300)
dataset200= main_create_b('skeleton','L',batch_size = 1,directory='/neurospin/dico/adneves/benchmark/200/Lskeleton_200')
dataset500= main_create_b('skeleton','L',batch_size = 1,directory='/neurospin/dico/adneves/benchmark/500/Lskeleton_500')
dataset800= main_create_b('skeleton','L',batch_size = 1,directory='/neurospin/dico/adneves/benchmark/800/Lskeleton_800')
datasetraw= main_create_b('skeleton','L',batch_size = 1,directory='/neurospin/dico/adneves/benchmark/raw/Lskeleton_raw')
list200=np.empty((len(dataset200),1728))
list500=np.empty((len(dataset500),1728))
list800=np.empty((len(dataset800),1728))
listraw=np.empty((len(datasetraw),1728))
df_subset200=dict()
df_subset500=dict()
df_subset800=dict()
df_subsetraw=dict()
list_name200=[]
list_name500=[]
list_name800=[]
list_nameraw=[]

for i,batch_test in enumerate(dataset200):
    real_imgs = Variable(batch_test[0].type(torch.Tensor))
    #list_name200 += [str(batch_test[1])+'_200']
    list_name200 += ['200']
    z = encoder(real_imgs).detach().squeeze(0).numpy()
    list200[i]=z
    #np.append(list200,np.nan_to_num(z))

for i,batch_test in enumerate(dataset500):
    real_imgs = Variable(batch_test[0].type(torch.Tensor))
    #list_name500 += [str(batch_test[1])+'_500']
    list_name500 += ['500']
    z = encoder(real_imgs).detach().squeeze(0).numpy()
    list500[i]=z

for i,batch_test in enumerate(dataset800):
    real_imgs = Variable(batch_test[0].type(torch.Tensor))
    #list_name800 += [str(batch_test[1])+'_800']
    list_name800 += ['800']
    z = encoder(real_imgs).detach().squeeze(0).numpy()
    list800[i]=z

for i,batch_test in enumerate(datasetraw):
    real_imgs = Variable(batch_test[0].type(torch.Tensor))
    list_nameraw += ['0']

    z = encoder(real_imgs).detach().squeeze(0).numpy()
    listraw[i]=z

tsne_results200 = tsne.fit_transform(list200)
df_subset200['tsne-2d-one'] = tsne_results200[:,0]
df_subset200['tsne-2d-two'] = tsne_results200[:,1]

tsne_results500 = tsne.fit_transform(list500)
df_subset500['tsne-2d-one'] = tsne_results500[:,0]
df_subset500['tsne-2d-two'] = tsne_results500[:,1]

tsne_results800 = tsne.fit_transform(list800)
df_subset800['tsne-2d-one'] = tsne_results800[:,0]
df_subset800['tsne-2d-two'] = tsne_results800[:,1]

tsne_resultsraw = tsne.fit_transform(listraw)
df_subsetraw['tsne-2d-one'] = tsne_resultsraw[:,0]
df_subsetraw['tsne-2d-two'] = tsne_resultsraw[:,1]

plt.figure(figsize=(16,10))

fig1=sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    palette=sns.color_palette("hls", 10),
    data=df_subset200,
    legend="full",
    alpha=0.3
)
fig2=sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    palette=sns.color_palette("hls", 10),
    edgecolor="black",
    data=df_subset500,
    legend="full",
    alpha=0.3
)
fig3=sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    palette=sns.color_palette("hls", 10),
    edgecolor="green",
    data=df_subset800,
    legend="full",
    alpha=0.3
)
fig4=sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    palette=sns.color_palette("hls", 10),
    edgecolor="red",
    data=df_subsetraw,
    legend="full",
    alpha=0.3
)

for k in range(len(dataset200)):
    plt.text(x=df_subset200['tsne-2d-one'][k], y=df_subset200['tsne-2d-two'][k] + 1, s = list_name200[k] )
for k in range(len(dataset500)):
    plt.text(x=df_subset500['tsne-2d-one'][k], y=df_subset500['tsne-2d-two'][k] + 1, s = list_name500[k])
for k in range(len(dataset800)):
    plt.text(x=df_subset800['tsne-2d-one'][k], y=df_subset800['tsne-2d-two'][k] + 1, s = list_name800[k])
for k in range(len(datasetraw)):
    plt.text(x=df_subsetraw['tsne-2d-one'][k], y=df_subsetraw['tsne-2d-two'][k] + 1, s = list_nameraw[k])

plt.show()
