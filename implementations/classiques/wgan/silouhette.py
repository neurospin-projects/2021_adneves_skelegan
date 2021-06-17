from sklearn.cluster import KMeans
import sklearn.metrics as sk
import argparse
import os
import numpy as np
import math
import sys
from yellowbrick.cluster import SilhouetteVisualizer
import matplotlib.pyplot as plt
import seaborn as sns
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

_, skel_train, skel_val, skel_test = main_create('skeleton','L',batch_size = 1, nb = 1000,adn=False, directory_base='/neurospin/dico/deep_folding_data/data/crops/STS_branches/nearest/original/Lskeleton')

list=np.empty((len(skel_test),1728))
df_subset=dict()
list_name1=[]

for i,batch_test in enumerate(skel_test):
    real_imgs = Variable(batch_test[0].type(torch.Tensor))
    list_name1 += [batch_test[1]]

    z = encoder(real_imgs).detach().squeeze(0).numpy()
    list[i]=z



fig, ax = plt.subplots(2, 2, figsize=(15,8))
for i in [2, 3, 4, 5]:


    km = KMeans(n_clusters=i, init='k-means++', random_state=42)
    q, mod = divmod(i, 2)

    visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q-1][mod])
    visualizer.fit(list)
plt.show()
