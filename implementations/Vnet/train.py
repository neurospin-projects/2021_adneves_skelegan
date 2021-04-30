#!/usr/bin/env python3

from local import *
import time
import argparse

import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader

import torchbiomed.loss as bioloss
import torchbiomed.utils as utils
from display_loss import *
import os
import sys
import math

import shutil

import setproctitle
from create_sets import *
import vnet
import make_graph
from functools import reduce
import operator
import torch

device = torch.device("cuda", index=0)

def max_search(input):
    _,_,x,y,z = input.shape
    max_list=[]
    for i in range(x):
        for j in range(y):
            for k in range(z):
                if input[0,0,i,j,k] not in max_list:
                    max_list += [int(input[0,0,i,j,k])]

    return 'max_list : ', max_list
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        nn.init.kaiming_normal(m.weight)
        m.bias.data.zero_()

def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)


def save_checkpoint(state, is_best, path, prefix, filename='checkpoint.pth.tar'):
    prefix_save = os.path.join('/home/ad265693/GAN/implementations/Vnet', path)
    name = prefix_save + '_' + filename
    print('saving ... ')
    torch.save(state, name)
    print('model saved to ' + name)
    if is_best:
        shutil.copyfile(name, prefix_save + '_model_best.pth.tar')


def inference(args, loader, model, transforms):
    src = args.inference
    dst = args.save

    model.eval()
    nvols = reduce(operator.mul, target_split, 1)
    # assume single GPU / batch size 1
    for data in loader:
        data, series, origin, spacing = data[0]
        shape = data.size()
        # convert names to batch tensor
        if args.cuda:
            data.pin_memory()
            data = data.cuda()
        data = Variable(data, volatile=True)
        output = model(data)
        _, output = output.max(1)
        output = output.view(shape)
        output = output.cpu()
        # merge subvolumes and save
        results = output.chunk(nvols)
        results = map(lambda var : torch.squeeze(var.data).numpy().astype(np.int16), results)
        volume = utils.merge_image([*results], target_split)
        print("save {}".format(series))
        utils.save_updated_image(volume, os.path.join(dst, series + ".mhd"), origin, spacing)


def noop(x):
    return x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=1)
    parser.add_argument('--dice', action='store_true')
    parser.add_argument('--ngpu', type=int, default=0)
    parser.add_argument('--nEpochs', type=int, default=10)
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('-i', '--inference', default='', type=str, metavar='PATH',
                        help='run inference on data set and save results')

    # 1e-8 works well for lung masks but seems to prevent
    # rapid learning for nodule masks
    parser.add_argument('--weight-decay', '--wd', default=1e-8, type=float,
                        metavar='W', help='weight decay (default: 1e-8)')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--save')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--opt', type=str, default='adam',
                        choices=('sgd', 'adam', 'rmsprop'))
    args = parser.parse_args()
    best_prec1 = 100.
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.save = args.save or 'work/vnet.base.{}'.format(datestr())
    nll = True
    if args.dice:
        nll = False
    weight_decay = args.weight_decay
    setproctitle.setproctitle(args.save)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    print("build vnet")
    model = vnet.VNet(elu=False, nll=nll)
    batch_size = args.ngpu*args.batchSz
    gpu_ids = range(args.ngpu)
    model = nn.parallel.DataParallel(model, device_ids=gpu_ids)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        model.apply(weights_init)

    if nll:
        train = train_nll
        test = test_nll
        class_balance = True
    else:
        train = train_dice
        test = test_dice
        class_balance = False

    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    if args.cuda:
        model = model.cuda()

    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)
    try:
        os.mkdir(join(args.save, 'images'))
    except:
        print ("dossier image déjà crée")


    if args.inference != '':
        if not args.resume:
            print("args.resume must be set to do inference")
            exit(1)
        kwargs = {'num_workers': 1} if args.cuda else {}
        src = args.inference
        dst = args.save
        inference_batch_size = args.ngpu
        root = os.path.dirname(src)
        images = os.path.basename(src)
        dataset = dset.LUNA16(root=root, images=images, transform=testTransform, split=target_split, mode="infer")
        loader = DataLoader(dataset, batch_size=inference_batch_size, shuffle=False, collate_fn=noop, **kwargs)
        inference(args, loader, model, trainTransform)
        return

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    print("loading training and test set")

    '''trainSet = dset.LUNA16(root='luna16', images=ct_images, targets=ct_targets,
                           mode="train", transform=trainTransform,
                           class_balance=class_balance, split=target_split, seed=args.seed, masks=masks)
    trainLoader = DataLoader(trainSet, batch_size=batch_size, shuffle=True, **kwargs)

    testLoader = DataLoader(
        dset.LUNA16(root='luna16', images=ct_images, targets=ct_targets,
                    mode="test", transform=testTransform, seed=args.seed, masks=masks, split=target_split),
        batch_size=batch_size, shuffle=False, **kwargs)
'''

    '''_, skel_train, skel_val, skel_test = main_create('skeleton','L',batch_size = args.batchSz, nb=1000)
    _, gw_train, gw_val, gw_test = main_create('gw','L',batch_size = args.batchSz,nb=1000)'''
    target_mean = 50
    bg_weight = target_mean / (1. + target_mean)
    fg_weight = 1. - bg_weight
    class_weights = torch.FloatTensor([bg_weight, fg_weight])
    if args.cuda:
        class_weights = class_weights.cuda()

    if args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=1e-1,
                              momentum=0.99, weight_decay=weight_decay)
    elif args.opt == 'adam':
        optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), weight_decay=weight_decay)

    err_best = 100.
    loss = []
    for epoch in range(1, args.nEpochs + 1):
        _, skel_train, skel_val, skel_test = main_create('skeleton','L',batch_size = args.batchSz, nb=1000)
        _, gw_train, gw_val, gw_test = main_create('gw','L',batch_size = args.batchSz,nb=1000)
        adjust_opt(args.opt, optimizer, epoch)
        loss_e = train(args, epoch, model,gw_train, skel_train, optimizer, class_weights)
        loss += loss_e
        err = test(args, epoch,skel_test, gw_test, model, optimizer, class_weights)
        is_best = False
        if err < best_prec1:
            is_best = True
            best_prec1 = err
        save_checkpoint({'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'best_prec1': best_prec1},
                        is_best, args.save, "vnet")
    display_loss_norm(loss)



def train_nll(args, epoch, model, gw_train, skel_train, optimizer, weights):
    model.train()
    nProcessed = 0
    nTrain = len(gw_train)
    batch_idx = 0
    list_loss_ep = []
    for batch_skel, batch_gw in zip(skel_train, gw_train):
        data = Variable(batch_skel[0].type(torch.Tensor)).to(device, dtype=torch.float32)
        target = Variable(batch_gw[0].type(torch.Tensor)).to(device, dtype=torch.long)
        target_size = target.shape
        target_flat= target.view(target.numel())
        optimizer.zero_grad()
        output = model(data)
        pred = output.data.max(1)[1]
        '''for item in output:
            max= torch.argmax(item)
            item = [0,0,0]
            item[max] = 1'''
        #loss = F.nll_loss(output, target, weight=weights)
        loss_t = nn.CrossEntropyLoss()
        loss= loss_t(output, target_flat)
            #make_graph.save('/tmp/t.dot', loss.creator); assert(False)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('enregistrement photo : ', batch_idx)
            image= pred.view(target_size)
            image = image.type(torch.float32)
            target_im = target.type(torch.float32)
            save_image(data[0,0,50,:,:], "/home/ad265693/GAN/implementations/Vnet/%s/%s/%s.png" % (args.save,"images","data" + str(epoch) + '_' + str(batch_idx)), nrow=5, normalize=True)
            save_image(target_im[0,0,50,:,:], "/home/ad265693/GAN/implementations/Vnet/%s/%s/%s.png" % (args.save,"images","target" + str(epoch) + '_' + str(batch_idx)), nrow=5, normalize=True)
            save_image(image[0,0,50,:,:], "/home/ad265693/GAN/implementations/Vnet/%s/%s/%s.png" % (args.save,"images","gw" + str(epoch) + '_' + str(batch_idx)), nrow=5, normalize=True)
            list_loss_ep += [loss]

        nProcessed += len(data)
        incorrect = pred.ne(target_flat.data).cpu().sum()
        err = 100.*incorrect/target.numel()
        partialEpoch = epoch + batch_idx / len(gw_train) - 1
        print('Epoque: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tError: {:.3f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(gw_train),
            loss, err))

        batch_idx += 1
    return list_loss_ep


def test_nll(args, epoch, skel_test, gw_test, model, optimizer, weights):
    print('go le test')
    model.eval()
    test_loss = 0
    dice_loss = 0
    incorrect = 0
    numel = 0
    batch_idx = 0
    for batch_skel, batch_gw in zip(skel_test, gw_test):
        print('testing batch : ', batch_idx)
        data_test = Variable(batch_skel[0].type(torch.torch.Tensor)).to(device, dtype=torch.float32)
        target_test = Variable(batch_gw[0].type(torch.torch.Tensor)).to(device, dtype=torch.long)
        target_flat= target_test.view(target_test.numel())
        target_size = target_test.shape
        optimizer.zero_grad()
        output = model(data_test)
        pred = output.data.max(1)[1]
        #loss = F.nll_loss(output, target, weight=weights)
        loss_t = nn.CrossEntropyLoss()
        numel += target_test.numel()
        test_loss += loss_t(output, target_flat)
        incorrect += pred.ne(target_flat.data).cpu().sum()
        batch_idx += 1

    test_loss /= len(gw_test)  # loss function already averages over batch size
    dice_loss /= len(gw_test)
    err = 100.*incorrect/numel
    print('\nTest set: Average loss: {:.4f}, Error: {}/{} ({:.3f}%) Dice: {:.6f}\n'.format(
        test_loss, incorrect, numel, err, dice_loss))

    return err


def adjust_opt(optAlg, optimizer, epoch):
    if optAlg == 'sgd':
        if epoch < 150:
            lr = 1e-1
        elif epoch == 150:
            lr = 1e-2
        elif epoch == 225:
            lr = 1e-3
        else:
            return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

if __name__ == '__main__':
    main()
