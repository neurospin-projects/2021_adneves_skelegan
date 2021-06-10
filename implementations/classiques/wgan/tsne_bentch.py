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
list=np.empty((4,1728))
list2=np.empty((4,1728))
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
dataset= main_create_b('skeleton','L',batch_size = 1,directory='/neurospin/dico/adneves/pickles/Lskeleton_benchmark')
dataset2= main_create_b('skeleton','L',batch_size = 1,directory='/neurospin/dico/adneves/pickles/Lskeleton_normal')
df_subset=dict()
df_subset2=dict()
list_name1=[]
list_name2=[]
for batch_test in dataset:
    real_imgs = Variable(batch_test[0].type(torch.Tensor))
    list_name1 += [batch_test[1]]

    z = encoder(real_imgs).detach().squeeze(0).numpy()
    np.append(list,z)
for batch_test in dataset2:
    real_imgs = Variable(batch_test[0].type(torch.Tensor))
    list_name2 += [batch_test[1]]

    z = encoder(real_imgs).detach().squeeze(0).numpy()
    np.append(list2,z)
tsne_results = tsne.fit_transform(list)
df_subset['tsne-2d-one'] = tsne_results[:,0]
df_subset['tsne-2d-two'] = tsne_results[:,1]

tsne_results2 = tsne.fit_transform(list2)
df_subset2['tsne-2d-one'] = tsne_results2[:,0]
df_subset2['tsne-2d-two'] = tsne_results2[:,1]

plt.figure(figsize=(16,10))

fig1=sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    palette=sns.color_palette("hls", 10),
    data=df_subset,
    legend="full",
    alpha=0.3
)
fig2=sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    palette=sns.color_palette("hls", 10),
    edgecolor="black",
    data=df_subset2,
    legend="full",
    alpha=0.3
)

for k in range(4):
    plt.text(x=df_subset['tsne-2d-one'][k], y=df_subset['tsne-2d-two'][k] + 20, s = list_name1[k])
for k in range(4):
    plt.text(x=df_subset2['tsne-2d-one'][k], y=df_subset2['tsne-2d-two'][k] + 20, s = list_name2[k])
plt.show()
