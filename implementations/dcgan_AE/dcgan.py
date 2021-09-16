import argparse
import os
import numpy as np
import math
from dcgan_model import *
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
parser.add_argument("--img_size", type=int, default=192, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument("--save")
parser.add_argument("--test", type=bool, default=False)
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
opt = parser.parse_args()
print(opt)

if os.path.exists(join('/neurospin/dico/adneves/dcgan_res/',opt.save)):
    shutil.rmtree(join('/neurospin/dico/adneves/dcgan_res/',opt.save))
os.makedirs(join('/neurospin/dico/adneves/dcgan_res/',opt.save), exist_ok=True)
try:
    os.mkdir(join('/neurospin/dico/adneves/dcgan_res/',opt.save, 'images'))
except:
    print ("dossier image déjà crée")

img_shape = (opt.channels, opt.img_size, opt.img_size,opt.img_size)

cuda = True if torch.cuda.is_available() else False


def save_checkpoint(epoch,state, filename='checkpoint.pth.tar'):
    prefix_save = os.path.join('/neurospin/dico/adneves/dcgan_res/', opt.save)
    name = prefix_save + '_' + filename
    print('saving ... ')
    torch.save(state, name)
    print('model saved to ' + name)
    shutil.copyfile(name, prefix_save + '_' + str(epoch) + '_model_best.pth.tar')


model = dcGAN(opt.latent_dim, img_shape, opt.batch_size).to(device)

if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> chargement checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> chargé à l'époque n° (epoch {})"
              .format( checkpoint['epoch']))
    else:
        print("=> pas de checkpoint trouvé '{}'".format(opt.resume))

# Loss function

# Optimizers
optimizer_E = torch.optim.Adam(model.Encoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_G = torch.optim.Adam(model.Generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(model.Discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
W = Tensor(3)
W[0],W[1],W[2] = 1,1,1

criterion_pixel =torch.nn.CrossEntropyLoss(weight = W)
adversarial_loss = torch.nn.BCELoss()
print('  + Number of encoder params: {}'.format(
        sum([p.data.nelement() for p in model.Encoder.parameters()])))
# ----------
#  Training
# ----------

loss_gen= []
n_epoch = 0
for epoch in range(1, opt.n_epochs + 1):
    model.train()
    i = 0
    if opt.resume:
        n_epoch = checkpoint['epoch']
    _, skel_train, skel_val, skel_test = main_create('skeleton','L',batch_size = opt.batch_size, nb = 1000)

    for batch_skel in skel_train:
        torch.cuda.empty_cache()
        target = Variable(batch_skel[0].type(torch.Tensor)).to(device, dtype=torch.float32)
        target_long = Variable(batch_skel[0].type(torch.Tensor)).to(device, dtype=torch.long)

        gen_img,gen_pred= model(target)
        # Loss measures generator's ability to fool the discriminator
        e_loss =criterion_pixel(gen_img, target_long.view(target_long.numel()))
        g_loss = e_loss

        e_loss.backward(retain_graph=True)
        g_loss.backward(retain_graph=True)
        optimizer_E.step()
        optimizer_G.step()

        loss_gen += [g_loss.item()]
        i += 1

        print(
            "[Epoque %d/%d] [Batch %d/%d] [G loss: %f] [E loss: %f]"
            % (n_epoch + epoch, opt.n_epochs+n_epoch, i, len(skel_train), g_loss.item(), e_loss.item())
        )

        batches_done = epoch * len(skel_train) + i
        if batches_done % opt.sample_interval == 0:
            save_image(target[0,0,30,:,:], "/neurospin/dico/adneves/dcgan_res/%s/%s/%s.png" % (opt.save,"images","data" + str(epoch) + '_' + str(batches_done)), nrow=5, normalize=True)
            save_image(gen_pred[0,0,30,:,:], "/neurospin/dico/adneves/dcgan_res/%s/%s/%s.png" % (opt.save,"images","target" + str(epoch) + '_' + str(batches_done)), nrow=5, normalize=True)
display_loss_norm(loss_gen)
save_checkpoint(epoch + n_epoch,{'epoch': n_epoch + epoch,
                     'state_dict': model.state_dict(),
                     })


if opt.test:
    model.eval()
    #### TEST
    print('début phase de test')
    loss_enc=0
    for batch_skel in skel_val:
        target = Variable(batch_skel[0].type(torch.Tensor)).to(device, dtype=torch.float32)
        target_long = Variable(batch_skel[0].type(torch.Tensor)).to(device, dtype=torch.long)

        gen_img,gen_pred= model(target)
        # Loss measures generator's ability to fool the discriminator
        e_loss = criterion_pixel(gen_img, target_long.view(target_long.numel()))

        loss_enc += e_loss.item()
    print('loss de reconstruction en test : ', loss_enc)

## génération de squelettes nouveaux
if opt.generation !=0:
    print('début phase de génération')
    gen_n = 0
    for new_im in range(opt.generation):
        z = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))
        gen_img = model.Generator(z)
        gen_pred_flat = gen_img.data.max(1)[1]
        gen_pred= gen_pred_flat.view(opt.batch_size,1,img_shape[1],img_shape[1],img_shape[1]).type(torch.float32)
        save_image(gen_pred[0,0,30,:,:], "/neurospin/dico/adneves/dcgan_res/%s/%s/%s.png" % (opt.save,"images","new_im_" + str(new_im)), nrow=5, normalize=True)
        gen_n += 1
        print("images générées %s/%s" % (gen_n,opt.generation))
print('fini !')
