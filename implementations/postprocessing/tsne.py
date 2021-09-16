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

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from utils.create_sets import *
import torch
import torch.nn as nn

''' Permet d'afficher un tsn-e en 2D ou 3D des cerceaux test sur un modèle choisis'''


parser = argparse.ArgumentParser()
'''Resume est le chemin vers le modèle que l'ont veut utiliser'''

parser.add_argument("--resume", default=None)
parser.add_argument("--nb_dim", type=int, default=2)
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = torch.device("cuda", index=0)
img_shape = (1, 80, 80,80)
twodim=(opt.nb_dim == 2)
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

_, skel_train, skel_val, skel_test = main_create('skeleton','L',batch_size = 1, nb = 1000,adn=False, directory_base='/neurospin/dico/data/deep_folding/data/crops/STS_branches/nearest/1mm/Lskeleton')

list=np.empty((len(skel_test),1728))
df_subset=dict()
list_name1=[]

for i,batch_test in enumerate(skel_test):
    real_imgs = Variable(batch_test[0].type(torch.Tensor))
    list_name1 += [batch_test[1]]

    z = encoder(real_imgs).detach().squeeze(0).numpy()
    list[i]=z

if twodim:
    tsne = TSNE(n_components=2, perplexity=20, n_iter=300)

    tsne_results = tsne.fit_transform(list)
    df_subset['tsne-2d-one'] = tsne_results[:,0]
    df_subset['tsne-2d-two'] = tsne_results[:,1]



    plt.figure(figsize=(16,10))

    fig1=sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        palette=sns.color_palette("hls", 10),
        data=df_subset,
        legend="full",
        alpha=0.3
    )

    for k in range(len(skel_test)):
        plt.text(x=df_subset['tsne-2d-one'][k], y=df_subset['tsne-2d-two'][k] + 0.2, s = list_name1[k])

    plt.show()
else:
    tsne = TSNE(n_components=3, perplexity=20, n_iter=300)
    tsne_results = tsne.fit_transform(list)
    x = tsne_results[:,0]
    y = tsne_results[:,1]
    z = tsne_results[:,2]


    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')



    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.scatter(x, y, z)


    '''for k in range(len(skel_test)):
        plt.text(x=x[k], y=y[k] + 0.2, s = list_name1[k])'''

    plt.show()
