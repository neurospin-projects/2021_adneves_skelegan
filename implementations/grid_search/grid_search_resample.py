import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image,make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import shutil
from create_sets2 import *
from display_loss import *
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
from torchsummary import summary

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = torch.device("cuda", index=0)
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--patience", type=int,default=20)
parser.add_argument("--valid", type=int,default=50)
parser.add_argument("--generation", default=0, type=int)
parser.add_argument("--lbd", type=float,default=1.)
parser.add_argument("--sulcus_weight", default=1,type=float)
parser.add_argument("--img_size", type=int, default=40, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=100, help="interval betwen image samples")
parser.add_argument("--resume", default=None)
opt = parser.parse_args()
print(opt)


img_shape = (opt.channels, opt.img_size, opt.img_size,opt.img_size)


range_n_critic=[1,2,3,4,5]
range_dim_latent=[200,300,500,750,1000]
range_lr=[0.0001,0.0005]
res = dict()
_, skel_train, skel_val, _ = main_create('skeleton','L',batch_size = opt.batch_size, nb = 1000,adn=False, directory_base='/neurospin/dico/data/deep_folding/data/crops/STS_branches/sulcus_based/2mm/Lskeleton')
_, _, _, skel_test = main_create('skeleton','L',batch_size =1, nb = 1000,adn=False, directory_base='/neurospin/dico/data/deep_folding/data/crops/STS_branches/sulcus_based/2mm/Lskeleton')
for n_critic in range_n_critic:
    for dim_latent in range_dim_latent:
        for lr in range_lr:
            print(" Grid Search state : n_critic = %d, dim_latent=%d, learning_rate(adam)= %f " %(n_critic,dim_latent,lr))
            print('len de skel train', len(skel_train))
            name_dir="critic%d_latent%d_lr%f" %(n_critic,dim_latent,lr)
            if os.path.exists(join('/neurospin/dico/adneves/grid_search',name_dir)):
                shutil.rmtree(join('/neurospin/dico/adneves/grid_search',name_dir))
            os.makedirs(join('/neurospin/dico/adneves/grid_search',name_dir), exist_ok=True)
            try:
                os.mkdir(join('/neurospin/dico/adneves/grid_search',name_dir, 'images'))
            except:
                print ("dossier image déjà crée")


            class Encoder(nn.Module):
                def __init__(self,batch_size,latent_dim, in_channels=1, dim=8, n_downsample=2):
                    super(Encoder, self).__init__()
                    self.batch_size = batch_size
                    self.latent_dim=latent_dim
                    self.l1 = nn.Sequential(nn.Linear(2048, self.latent_dim))
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
                    z=z.view((z.shape[0],-1))
                    z=self.l1(z)
                    return z

            class Generator(nn.Module):
                def __init__(self, latent_dim,img_shape):
                    super(Generator, self).__init__()
                    self.latent_dim =latent_dim
                    self.init_size = img_shape[1] // 4
                    self.l1 = nn.Sequential(nn.Linear(self.latent_dim, 128 * self.init_size ** 3))

                    self.conv_blocks = nn.Sequential(
                        nn.BatchNorm3d(128),
                        nn.Upsample(scale_factor=2),
                        nn.Conv3d(128, 128, 3, stride=2, padding=1),
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
                        nn.Linear(64000, 512),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Linear(512, 256),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Linear(256, 1),
                    )

                def forward(self, img):
                    img_flat = img.view(img.shape[0], -1).to(torch.float32)
                    validity = self.model(img_flat)
                    return validity

            W = torch.Tensor(3).to(device, dtype=torch.float32)
            W[0],W[1],W[2] = 1,1,opt.sulcus_weight

            criterion_pixel =torch.nn.CrossEntropyLoss(weight = W)
            # Loss weight for gradient penalty
            lambda_gp = 100
            valid_scores=[]
            # Initialize generator and discriminator
            encoder= Encoder(opt.batch_size,dim_latent).to(device, dtype=torch.float32)
            generator = Generator(dim_latent,img_shape).to(device, dtype=torch.float32)
            discriminator = Discriminator().to(device, dtype=torch.float32)
            '''
            summary(encoder, (opt.batch_size, 1,80,80,80))
            summary(generator, (opt.batch_size,opt.latent_dim))
            summary(discriminator, (opt.batch_size, 1,80,80,80))
            '''
            optimizer_E = torch.optim.Adam(generator.parameters(), lr=lr, betas=(opt.b1, opt.b2))
            optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(opt.b1, opt.b2))
            optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(opt.b1, opt.b2))



            def compute_gradient_penalty(D, real_samples, fake_samples):
                """Calculates the gradient penalty loss for WGAN GP"""
                # Random weight term for interpolation between real and fake samples
                alpha = Tensor(np.random.random((real_samples.shape[0], 1, 1, 1,1)))
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
                    encoder.load_state_dict(checkpoint['state_dict_enc'])
                    n_epoch = checkpoint['epoch']
                    print("=> chargé à l'époque n° (epoch {})"
                          .format( checkpoint['epoch']))
                else:
                    print("=> pas de checkpoint trouvé '{}'".format(opt.resume))


            loss_gen,loss_disc,loss_enc=[],[],[]
            i = 0
            #directory_base='/neurospin/dico/deep_folding_data/data/crops/STS_branches/nearest/original/Lskeleton'
            for epoch in range(1,opt.n_epochs+1):
                for batch_skel in skel_train:
                    # Configure input
                    torch.cuda.empty_cache()
                    real_imgs = Variable(batch_skel[0].type(torch.Tensor)).to(device, dtype=torch.float32)
                    # ---------------------
                    #  Train Discriminator
                    # ---------------------
                    optimizer_E.zero_grad()
                    optimizer_D.zero_grad()
                    # Sample noise as generator input
                    z = encoder(real_imgs).to(device, dtype=torch.float32)

                    # Generate a batch of images
                    fake_imgs = generator(z)
                    fake_maxed =fake_imgs.max(1)[1].to(torch.float32)

                        # Real images
                    real_validity = discriminator(real_imgs)
                    # Fake images
                    fake_validity = discriminator(fake_maxed)
                    # Gradient penalty
                    gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_maxed.data.unsqueeze(1))
                    # Adversarial loss
                    d_loss = -1*opt.lbd*torch.mean(real_validity) + opt.lbd*torch.mean(fake_validity) + lambda_gp * gradient_penalty

                    optimizer_G.zero_grad()

                        # Train the generator every n_critic steps
                    if i % n_critic == 0:

                        # -----------------
                        #  Train Generator
                        # -----------------

                        # Generate a batch of images
                        #fake_imgs = generator(z)
                        # Loss measures generator's ability to fool the discriminator
                        # Train on fake images
                        #fake_validity = discriminator(fake_imgs)

                        loss_CE=criterion_pixel(fake_imgs, real_imgs.squeeze(1).long())
                        g_loss = -opt.lbd*torch.mean(fake_validity) + 100*loss_CE
                        e_loss= 100*loss_CE

                        g_loss.backward( retain_graph=True)
                        e_loss.backward( retain_graph=True)
                        optimizer_G.step()
                        optimizer_E.step()
                        loss_disc += [d_loss.item()]
                        loss_gen += [g_loss.item()]
                        loss_enc += [e_loss.item()]

                        print(
                            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [E loss: %f]"
                            % (epoch, opt.n_epochs, i, (epoch+1)*len(skel_train), d_loss.item(), g_loss.item(),e_loss.item())
                        )
                    if i % opt.valid==0:
                        with torch.no_grad():
                            loss_enco=0
                            j=0
                            for batch_skel in skel_val:
                                real_imgs = Variable(batch_skel[0].type(torch.Tensor)).to(device, dtype=torch.float32)
                                z = encoder(real_imgs).to(device, dtype=torch.float32)

                                fake_imgs = generator(z)
                                fake_maxed =fake_imgs.max(1)[1].to(torch.float32)
                                # Loss measures generator's ability to fool the discriminator
                                loss_CE = criterion_pixel(fake_imgs, real_imgs.squeeze(1).long())
                                e_loss= 100*loss_CE
                                loss_enco += e_loss.item()
                                j += 1
                            valid_score=loss_enco/len(skel_val)
                            valid_scores += [valid_score]
                            print('loss de reconstruction en test : ', valid_score)


                    if i % opt.sample_interval == 0:
                        real_imgs1 = make_grid(real_imgs[0,0,10,:,:], nrow=5, normalize=True)
                        real_imgs2 = make_grid(real_imgs[0,0,20,:,:], nrow=5, normalize=True)
                        real_imgs3 = make_grid(real_imgs[0,0,30,:,:], nrow=5, normalize=True)
                        image_grid = torch.cat((real_imgs1, real_imgs2, real_imgs3), 1)
                        save_image(image_grid, "/neurospin/dico/adneves/grid_search/%s/%s/%s.png" % (name_dir,"images","target" + str(epoch) + '_' + str(i)), nrow=5, normalize=True)
                        fake_imgs1 = make_grid(fake_maxed[0,10,:,:], nrow=5, normalize=True)
                        fake_imgs2 = make_grid(fake_maxed[0,20,:,:], nrow=5, normalize=True)
                        fake_imgs3 = make_grid(fake_maxed[0,30,:,:], nrow=5, normalize=True)
                        fake_grid = torch.cat((fake_imgs1, fake_imgs2, fake_imgs3), 1)
                        save_image(fake_grid, "/neurospin/dico/adneves/grid_search/%s/%s/%s.png" % (name_dir,"images","fake_img" + str(epoch) + '_' + str(i)), nrow=5, normalize=True)
                    d_loss.backward( retain_graph=True)
                    optimizer_D.step()
                    i += 1
                    stop = (len(valid_scores) >= opt.patience) and (all( x <valid_scores[-1] for x in valid_scores[-1 * opt.patience:-1]))

                    if stop:
                        print('stopped at epoch %d' %(epoch+1))
                        break
                if stop:
                    break
            print('saving ... ')
            state={'epoch': n_epoch +epoch+1, 'state_dict_gen': generator.state_dict(),'state_dict_disc': discriminator.state_dict(),'state_dict_enc': encoder.state_dict()}
            name=join('/neurospin/dico/adneves/grid_search/', 'epoch_'+str(epoch+1)+'_'+ str(name_dir)+ '_checkpoint.pth.tar')
            torch.save(state, name)
            print('model saved to ' + name)
            res[('n_critic_'+str(n_critic), 'latent_dim_'+str(dim_latent), 'learning_rate_'+str(lr))] = (valid_scores[-1],epoch+1)
            print(res)
            ## génération de squelettes nouveaux
            if opt.generation !=0:
                print('début phase de génération')
                gen_n = 0
                for new_im in range(opt.generation):
                    z = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, dim_latent))))
                    fake_imgs = generator(z)
                    fake_maxed =fake_imgs.max(1)[1].to(torch.float32)
                    fake_imgs1 = make_grid(fake_maxed[0,10,:,:], nrow=5, normalize=True)
                    fake_imgs2 = make_grid(fake_maxed[0,20,:,:], nrow=5, normalize=True)
                    fake_imgs3 = make_grid(fake_maxed[0,30,:,:], nrow=5, normalize=True)
                    fake_grid = torch.cat((fake_imgs1, fake_imgs2, fake_imgs3), 1)
                    save_image(fake_grid, "/neurospin/dico/adneves/grid_search/%s/%s/%s.png" % (name_dir,"images","new_im_" + str(new_im)), nrow=5, normalize=True)
                    gen_n += 1
                    print("images générées %s/%s" % (gen_n,opt.generation))

            print('génération du cerveau moyen')
            latents=torch.empty((len(skel_test),dim_latent))
            print('len de skel test', len(skel_test))
            for i,batch_test in enumerate(skel_test):
                real_imgs = Variable(batch_test[0].type(torch.Tensor)).to(device, dtype=torch.float32)
                z = encoder(real_imgs).detach().to(device, dtype=torch.float32)
                latents[i]=z.squeeze(0)
            sample_sum=torch.sum(latents,0)/len(skel_test)
            sample_sum=sample_sum.unsqueeze(0).to(device, dtype=torch.float32)
            gen_s=generator(sample_sum).to(device, dtype=torch.float32)
            gen_max =gen_s.max(1)[1].to(torch.float32).cpu().detach().numpy()
            #save_image(gen_max[0,30,:,:], "/neurospin/dico/adneves/wgan_gp/%s/%s.png" % (opt.save,"vect_moyen"), nrow=5, normalize=True)
            np.save("/neurospin/dico/adneves/grid_search/%s/vectmoyen.npy" % (name_dir), gen_max)
