# /usr/bin/env python3
# coding: utf-8
"""
The aim of this script is to create pytorch dataloaders from MRIs saved as
numpy arrays in a .pickle.
"""
import os
import time
from datetime import date
import argparse
import pandas as pd
import itertools
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from os.path import join
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchio as tio

from pynet_transforms import *
import save_results


class TensorDataset_skeleton():
    """Custom dataset that includes image file paths.
    Apply different transformations to data depending on the type of input.
    IN: data_tensor: tensor containing MRIs as numpy arrays
        filenames: list of subjects' IDs
        skeleton: boolean, whether input is skeleton images or not
    OUT: tensor of [batch, sample, subject ID]
    """
    def __init__(self, data_tensor, filenames):
        self.data_tensor = data_tensor
        self.transform = True
        self.nb_train = len(filenames)
        self.filenames = filenames

    def __len__(self):
        return(self.nb_train)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data_tensor[idx]
        file = self.filenames[idx]
        #self.transform1 = DownsampleTensor(scale=2)
        #self.transform1 = Padding([1, 192, 192, 192], fill_value=11)
        self.transform1 = Padding([1, 80, 80, 80], fill_value=11)
        #self.transform2 = Padding([1, 96, 96, 96], fill_value=0)
        sample = self.transform1(sample)
        values=[0,11,60]
        _,x,y,z = sample.shape
        sample[sample == 0] = 0 # inside the brain
        sample[sample == 11] = 1 # sulci
        sample[sample > 1] = 2 # out of the brain
        #sample = self.transform2(sample)
        tuple_with_path = (sample, file)
        return tuple_with_path

class TensorDataset_gw():
    """Custom dataset that includes image file paths.
    Apply different transformations to data depending on the type of input.
    IN: data_tensor: tensor containing MRIs as numpy arrays
        filenames: list of subjects' IDs
        skeleton: boolean, whether input is skeleton images or not
    OUT: tensor of [batch, sample, subject ID]
    """
    def __init__(self, data_tensor, filenames):
        self.data_tensor = data_tensor
        self.transform = True
        self.nb_train = len(filenames)
        self.filenames = filenames

    def __len__(self):
        return(self.nb_train)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data_tensor[idx]
        file = self.filenames[idx]
        self.transform1 = DownsampleTensor(scale=2)
        #self.transform2 = Padding([1, 192, 192, 192], fill_value=0)
        self.transform2 = Padding([1, 96, 96, 96], fill_value=0)
        print(sample.shape)
        sample = self.transform1(sample)
        print(sample.shape)
        _,x,y,z = sample.shape
        for i in range(x):
            for j in range(y):
                for k in range(z):
                    distances = [abs(sample[0,i,j,k]), abs(sample[0,i,j,k]-100), abs(sample[0,i,j,k] - 200) ]
                    sample[0,i,j,k] = np.argmin(distances)
        sample = self.transform2(sample)
        tuple_with_path = (sample, file)
        return tuple_with_path



class SkeletonDataset():
    """Custom dataset for skeleton images that includes image file paths.
    dataframe: dataframe containing training and testing arrays
    filenames: optional, list of corresponding filenames
    Works on CPUs
    """
    def __init__(self, dataframe, filenames=None):
        self.df = dataframe
        if filenames:
            self.filenames = filenames
            self.df = self.df.T
        else:
            self.filenames = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.filenames:
            filename = self.filenames[idx]
            sample = self.df.iloc[idx][0]
        else:
            filename = self.df.iloc[idx]['ID']
            sample = self.df.iloc[idx][0]

        fill_value = 1
        sample = NormalizeSkeleton(sample)()
        '''self.transform = transforms.Compose([Downsample(scale=2),
                         Padding([1, 40, 40, 40], fill_value=fill_value)
                         ])
        sample = self.transform(sample)'''
        tuple_with_path = (sample, filename)
        return tuple_with_path


class AugDatasetTransformer(torch.utils.data.Dataset):
    """
    Custom dataset that apply data augmentation on a dataset processed
    through TensorDataset class.
    Transformations are performed on CPU.
    """
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __getitem__(self, index):
        img, filename = self.base_dataset[index]
        if np.random.rand() > 0.6:
            self.angle = np.random.randint(-90, 90)
            img = np.expand_dims(rotate(img[0], angle=self.angle, reshape=False, cval=1, order=1), axis=0)
        return img, filename

    def __len__(self):
        return len(self.base_dataset)


def create_hcp_benchmark(side, benchmark, directory, batch_size, handedness=1):
    """
    Creates datasets from HCP data and depending on dataset split of benchmark
    generation (cf anatomist_tools.benchmark_generation module)
    /!\ ONLY DIFFERENCE FROM create_hcp_sets function is only that it creates
    sets from benchmark split.
    IN: side: str, 'right' or 'left'
        handedness: int, 1 if right handed, 2 if left handed
        directory: str, folder in which save the results
        batch_size: int, size of training batches
    OUT: root_dir,
         dataset_train_loader,
         dataset_val_loader,
         dataset_test_loader
    """
    date_exp = date.today().strftime("%d%m%y")

    train_list = pd.read_csv('/neurospin/dico/lguillon/mic21/anomalies_set/dataset/benchmark' + str(benchmark) + '/0_Lside/train.csv')
    train_list = train_list.rename(columns={"0":"Subject"})

    loss_type = 'CrossEnt'
    root_dir = directory + side + '_hemi_skeleton_' + date_exp + '_' +loss_type + '_' + str(handedness) + '_2classes/'
    print(root_dir)
    save_results.create_folder(root_dir)

    data_dir = '/neurospin/dico/lguillon/skeleton/sts_crop/'

    if handedness == 1:
        input_data = side + '_hemi_rightH_sts_crop_skeleton'
    else:
        input_data = side + '_hemi_leftH_sts_crop_skeleton'
    print(input_data)
    tmp = pd.read_pickle(data_dir + input_data +'.pkl')
    train = pd.merge(tmp, train_list.Subject.astype(str), on='Subject')
    train = train.reset_index(drop=True)

    hcp_dataset_train = SkeletonDataset(dataframe=train)

    # Split training set into train, val and test
    partition = [0.8,0.2, 0]
    print([round(i*(len(hcp_dataset_train))) for i in partition])
    train_set, val_set, test_set = torch.utils.data.random_split(hcp_dataset_train,
                         [round(i*(len(hcp_dataset_train))) for i in partition])

    # Data Augmentation application
    train_set = AugDatasetTransformer(train_set)
    #val_set = AugDatasetTransformer(val_set)
    #test_set  = AugDatasetTransformer(test_set)

    dataset_train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                shuffle=True, num_workers=8)
    dataset_val_loader = torch.utils.data.DataLoader(val_set, shuffle=True,
                                                          num_workers=0)
    dataset_test_loader = torch.utils.data.DataLoader(test_set, shuffle=True,
                                                              num_workers=0)

    return root_dir, dataset_train_loader, dataset_val_loader, dataset_test_loader


def create_benchmark_test(benchmark, side, handedness=1):
    """
    Creates test datasets from benchmark of altered skeletons
    (cf anatomist_tools.benchmark_generation module)
    IN: benchmark: int, number of benchmark
        handedness: int, 1 if right handed, 2 if left handed
    OUT: dataset_test_abnor_loader
    """
    data_dir = '/neurospin/dico/lguillon/mic21/anomalies_set/dataset/benchmark' + str(benchmark) + '/'

    input_data = 'abnormal_skeleton_' + side
    print(input_data)
    tmp = pd.read_pickle(data_dir + input_data +'.pkl')
    filenames = list(tmp.columns)

    benchmark_dataset = SkeletonDataset(dataframe=tmp, filenames=filenames)

    benchmark_loader = torch.utils.data.DataLoader(benchmark_dataset, batch_size=1,
                                                    shuffle=True, num_workers=0)

    return benchmark_loader


def create_hcp_sets(input_type, side, directory, batch_size):
    """
    Creates datasets from HCP data
    IN: skeleton: boolean, True if input is skeleton, False otherwise,
        side: str, 'right' or 'left'
        handedness: int, 0 if mixed ind, 1 if right handed, 2 if left handed
        directory: str, folder in which save the results
        batch_size: int, size of training batches
        weights: list, list of weights to apply to skeleton values
    OUT: root_dir: created directory where results will be stored
         dataset_train_loader, dataset_val_loader, dataset_test_loader: loaders
         that will be used for training and testing
    """
    date_exp = date.today().strftime("%d%m%y")
    root_dir= join(directory, date_exp)
    #save_results.create_folder(root_dir)
    tmp = pd.read_pickle(directory +'.pkl')
    filenames = list(tmp.columns)
    #print(tmp.loc[0,filenames[4]])
    tmp = torch.from_numpy(np.array([tmp.loc[0,file_name] for file_name in filenames]))
    #tmp = tmp.to('cuda')
    if input_type =='gw' :
        hcp_dataset = TensorDataset_gw(filenames=filenames, data_tensor=tmp)
    else:
        hcp_dataset = TensorDataset_skeleton(filenames=filenames, data_tensor=tmp)
    # Split training set into train, val and test
    partition = [0.7,0.2, 0.1]
    #print([round(i*(len(hcp_dataset))) for i in partition])
    #train_set, val_set, test_set = torch.utils.data.random_split(hcp_dataset, [round(i*(len(hcp_dataset))) for i in partition])
    train_set, val_set, test_set = torch.utils.data.random_split(hcp_dataset,[882,126,94])
    #train_set = AugDatasetTransformer(train_set)
    #val_set = AugDatasetTransformer(val_set)
    #test_set  = AugDatasetTransformer(test_set)

    dataset_train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                shuffle=True, num_workers=0)
    dataset_val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True,
                                                          num_workers=0)
    dataset_test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,shuffle=True,
                                                          num_workers=0)

    print("Dataset generated \n Size of training dataset :", len(dataset_train_loader))

    return root_dir, dataset_train_loader, dataset_val_loader, dataset_test_loader

def create_hcp_setsb(input_type, side, directory, batch_size):
    """
    Creates datasets from HCP data
    IN: skeleton: boolean, True if input is skeleton, False otherwise,
        side: str, 'right' or 'left'
        handedness: int, 0 if mixed ind, 1 if right handed, 2 if left handed
        directory: str, folder in which save the results
        batch_size: int, size of training batches
        weights: list, list of weights to apply to skeleton values
    OUT: root_dir: created directory where results will be stored
         dataset_train_loader, dataset_val_loader, dataset_test_loader: loaders
         that will be used for training and testing
    """
    date_exp = date.today().strftime("%d%m%y")
    root_dir= join(directory, date_exp)
    #save_results.create_folder(root_dir)
    tmp = pd.read_pickle(directory +'.pkl')
    filenames = list(tmp.columns)
    #print(tmp.loc[0,filenames[4]])
    tmp = torch.from_numpy(np.array([tmp.loc[0,file_name] for file_name in filenames]))
    #tmp = tmp.to('cuda')
    if input_type =='gw' :
        hcp_dataset = TensorDataset_gw(filenames=filenames, data_tensor=tmp)
    else:
        hcp_dataset = TensorDataset_skeleton(filenames=filenames, data_tensor=tmp)

    dataset= torch.utils.data.DataLoader(hcp_dataset, batch_size=batch_size,
                                                shuffle=False, num_workers=0)

    return dataset


def create_aims_sets(skeleton, side, handedness=0):
    """
    Creates datasets from AIMS data
    IN: skeleton: boolean, True if input is skeleton, False otherwise,
        side: str, 'right' or 'left'
        handedness: int, 0 if mixed ind, 1 if right handed, 2 if left handed
    OUT:
    """

    if handedness == 0:
        if skeleton:
            controls = pd.read_pickle('/neurospin/dico/lguillon/aims_detection/aims_crop/skeleton/{}_hemi/controls_qc_{}.pkl'.format(side, side))
            asd = pd.read_pickle('/neurospin/dico/lguillon/aims_detection/aims_crop/skeleton/{}_hemi/asd_qc_{}.pkl'.format(side, side))
            id_controls = pd.read_pickle('/neurospin/dico/lguillon/aims_detection/aims_crop/skeleton/{}_hemi/id_controls_qc_{}.pkl'.format(side, side))
            asd_id = pd.read_pickle('/neurospin/dico/lguillon/aims_detection/aims_crop/skeleton/{}_hemi/asd_id_qc_{}.pkl'.format(side, side))
        else:
            controls = pd.read_pickle('/neurospin/dico/lguillon/aims_detection/aims_crop/controls_qc_{}.pkl'.format(side))
            asd = pd.read_pickle('/neurospin/dico/lguillon/aims_detection/aims_crop/asd_qc_{}.pkl'.format(side))
            id_controls = pd.read_pickle('/neurospin/dico/lguillon/aims_detection/aims_crop/id_controls_qc_{}.pkl'.format(side))
            asd_id = pd.read_pickle('/neurospin/dico/lguillon/aims_detection/aims_crop/asd_id_qc_{}.pkl'.format(side))
        asd = asd.rename(columns={0:'mri'})
        asd_id = asd_id.rename(columns={0:'mri'})
        controls = controls.rename(columns={0:'mri'})
        id_controls = id_controls.rename(columns={0:'mri'})
    else:
        if handedness==1:
            hand = 'right'
        else:
            hand = 'left'
        if skeleton:
            #controls = pd.read_pickle('/neurospin/dico/lguillon/aims_detection/aims_crop/skeleton/{}_hemi/controls_qc_{}_{}H.pkl'.format(side, side, hand))
            #asd = pd.read_pickle('/neurospin/dico/lguillon/aims_detection/aims_crop/skeleton/{}_hemi/asd_qc_{}_{}H.pkl'.format(side, side, hand))
            #id_controls = pd.read_pickle('/neurospin/dico/lguillon/aims_detection/aims_crop/skeleton/{}_hemi/id_controls_qc_{}_{}H.pkl'.format(side, side, hand))
            #asd_id = pd.read_pickle('/neurospin/dico/lguillon/aims_detection/aims_crop/skeleton/{}_hemi/asd_id_qc_{}_{}H.pkl'.format(side, side, hand))

            controls = pd.read_pickle('/home_local/lg261972/data/data_handedness/{}_hemi/controls_qc_{}_{}H.pkl'.format(side, side, hand))
            asd = pd.read_pickle('/home_local/lg261972/data/data_handedness/{}_hemi/asd_qc_{}_{}H.pkl'.format(side, side, hand))
            id_controls = pd.read_pickle('/home_local/lg261972/data/data_handedness/{}_hemi/id_controls_qc_{}_{}H.pkl'.format(side, side, hand))
            asd_id = pd.read_pickle('/home_local/lg261972/data/data_handedness/{}_hemi/asd_id_qc_{}_{}H.pkl'.format(side, side, hand))
        else:
            controls = pd.read_pickle('/neurospin/dico/lguillon/aims_detection/aims_crop/{}_hemi/controls_qc_{}_{}H.pkl'.format(side, side, hand))
            asd = pd.read_pickle('/neurospin/dico/lguillon/aims_detection/aims_crop/{}_hemi/asd_qc_{}_{}H.pkl'.format(side, side, hand))
            id_controls = pd.read_pickle('/neurospin/dico/lguillon/aims_detection/aims_crop/{}_hemi/id_controls_qc_{}_{}H.pkl'.format(side, side, hand))
            asd_id = pd.read_pickle('/neurospin/dico/lguillon/aims_detection/aims_crop/{}_hemi/asd_id_qc_{}_{}H.pkl'.format(side, side, hand))

    filenames_asd = asd['ID']
    asd = torch.from_numpy(np.array([asd.mri.values[k] for k in range(len(asd))]))
    asd = asd.to('cuda')
    asd_dataset = TensorDataset(filenames=filenames_asd, data_tensor=asd,
                                skeleton=skeleton, vae=False)
    asd_dataset = torch.utils.data.DataLoader(asd_dataset, shuffle=True,
                                              num_workers=0, batch_size=1)

    filenames_controls = controls['ID']
    controls = torch.from_numpy(np.array([controls.mri.values[k] for k in range(len(controls))]))
    controls = controls.to('cuda')
    controls_dataset = TensorDataset(filenames=filenames_controls, data_tensor=controls,
                                    skeleton=skeleton, vae=False)
    controls_dataset = torch.utils.data.DataLoader(controls_dataset, shuffle=True,
                                              num_workers=0, batch_size=1)

    filenames_id_controls = id_controls['ID']
    id_controls = torch.from_numpy(np.array([id_controls.mri.values[k] for k in range(len(id_controls))]))
    id_controls = id_controls.to('cuda')
    id_controls_dataset = TensorDataset(filenames=filenames_id_controls, data_tensor=id_controls,
                                        skeleton=skeleton, vae=False)
    id_controls_dataset = torch.utils.data.DataLoader(id_controls_dataset, shuffle=True,
                                              num_workers=0, batch_size=1)

    filenames_asd_id = asd_id['ID']
    asd_id = torch.from_numpy(np.array([asd_id.mri.values[k] for k in range(len(asd_id))]))
    asd_id = asd_id.to('cuda')
    asd_id_dataset = TensorDataset(filenames=filenames_asd_id, data_tensor=asd_id,
                                   skeleton=skeleton, vae=False)
    asd_id_dataset = torch.utils.data.DataLoader(asd_id_dataset, shuffle=True,
                                              num_workers=0, batch_size=1)

    return asd_dataset, controls_dataset, id_controls_dataset, asd_id_dataset

def main_create(input_type,side, batch_size,nb,directory_base='/neurospin/dico/adneves/pickles/',adn=True) :
    #directory_base ='/neurospin/dico/adneves/output/'
    if adn:
        return create_hcp_sets(input_type=input_type, side=side, directory = join(directory_base, side + '_'+
    input_type, str(nb) + '_' + side + input_type) , batch_size=batch_size)
    else:
        return create_hcp_sets(input_type=input_type, side=side, directory = directory_base, batch_size=batch_size)

def main_create_b(input_type,side, batch_size,directory) :
    #directory_base ='/neurospin/dico/adneves/output/'
    return create_hcp_setsb(input_type='skeleton', side=side, directory = directory , batch_size=1)
