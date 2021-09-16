# A PyTorch implementation of V-Net

Vnet is a [PyTorch](http://pytorch.org/) implementation of the paper
[V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation](https://arxiv.org/abs/1606.04797)
by Fausto Milletari, Nassir Navab, and Seyed-Ahmad Ahmadi. \

![alt text](https://github.com/antdasneves/GAN/blob/main/vnet.png?raw=true)

It is used here to transform a brain MRI image in skeleton form, depicting only the sulci, into a grey-white image which conveys more information.


![alt text](https://github.com/antdasneves/GAN/blob/main/hcp.png?raw=true)

## Dependencies
* matplotlib==3.3.1
* numpy==1.18.0
* pandas==1.1.5
* pickleshare==0.7.5
* Pillow==8.1.2
* pyparsing==2.4.7
* scikit-image==0.15.0
* scikit-learn==0.21.3
* scipy==1.1.0
* seaborn==0.11.1
* setproctitle==1.2.2
* skorch==0.10.0
* tensorboard==1.13.1
* tensorflow==1.13.1
* torch==1.8.0+cu111
* torch-summary==1.4.5
* torchio==0.18.30
* torchvision==0.9.0+cu111


## Repositories
Where the Vnet code is, with the corresponding codes with the tools to explore its latent space : \
/home/ad265693/GAN/implementations/Vnet/

Where the data used is, corresponding to 1000 left brain hemispheres of different subjects: \
* For skeleton images: \
/neurospin/dico/adneves/pickles/L_skeleton/1000_Lskeleton.pkl \

* For grey-white images: \
/neurospin/dico/adneves/pickles/L_gw/1000_Lgw.pkl



The image generated and models are made to be saved in /neurospin/dico/adneves/
This path can be modified in the code easily

The Vnet uses the createsetsVnet.py code to create its training and validation database.
It can be found in /home/ad265693/dl_tools/

## How to use the script

My code have been used within Kraken and an virtual environment.\
When in the correct repository (/Vnet/), the following command will make the Vnet work:

~~~
python3 train.py --batchSz (batch size) --ngpu (number of GPU used) --nEpochs (number of epochs) --save (name of the repository were the results will be saved)
~~~ 
