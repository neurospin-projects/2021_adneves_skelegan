# GAN Gradient Penalty
The proposed codes are part of a deep learning projet conducted on brain sulci . The goal is, thanks to the wGAN Gradient Penalty model, to learn the human variability of sulci. We work on the HCP database (http://www.humanconnectomeproject.org/), MRI images previously processed through the Morphologist pipeline (https://brainvisa.info/web/morphologist.html).

![alt text](https://github.com/antdasneves/GAN/blob/main/schemaGan.png?raw=true)


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
* skorch==0.10.0
* tensorboard==1.13.1
* tensorflow==1.13.1
* torch==1.8.0+cu111
* torch-summary==1.4.5
* torchio==0.18.30
* torchvision==0.9.0+cu111


## Repositories
Where the wgan-gp code is, with the corresponding codes with the tools to explore its latent space : \
/home/ad265693/GAN/implementations/classiques/wgan/

Where the data used is, corresponding to crops of the s.t.s region: \
/neurospin/dico/data/deep_folding/data/crops/STS_branches/nearest/original/Lskeleton

The image generated and models are made to be saved in /neurospin/dico/adneves/
This path can be modified in the code easily

The wGAN-gp uses the createsets.py code to create its training and validation database.
It can be found in /home/ad265693/dl_tools/

## How to use the scripts

My code have been used within Kraken and an virtual environment.\
When in the correct repository (/wgan), the following command will make the wgan-p work:

~~~
python3 wgan_gp.py --save (name of save repository) --n_epochs (number of epochs) --latent_dim (number of latent dimensions) --img_size 80 --sample_interval (interval of sampling images) --batch_size (batch size) --n_critic (training ration generator/discriminator) --sulcus_weight (weight of sulci in the cross entropy) --resume (Optional : path to a model saved in '.pth.tar', will resume the training)
~~~

## Exploration of the latent space
Here, you can find codes allowing to explore and assess the quality of a model produced by the wGAN-gp

For the exploration_ligne.py script, there are 4 possible usages:

* Interpolation between two brains\
~~~
python3 exploration_ligne.py --ligne True --resume (path to the model) --save (saving repositories of image sampled) --nb_samples (number of points interpolated)
~~~

* Feature ranking\
This option allows us to create a dictionary ranking each dimension of the model by importance 

~~~
python3 exploration_ligne.py --resume (path to model evaluated) --feature_rank (number of dimensions)
~~~

* Average Brain\
Will save in .npy the average brain by averaging for each dimension, the code of all test brains encoded, then will generate with the trained generator the resulting brain image in 3D.
~~~
python3 exploration_ligne.py --sum_dim True --resume (path to the model) --save (saving repositories of image sampled)
~~~

* Dimension exploration\
For a chosen dimension, and chosen boundaries, will sample images of encoded brains varying acording the dimension.

~~~
python3 exploration_ligne.py --resume (path to the model) --save (saving repositories of image sampled) --nb_samples (number of points interpolated) --expl_dim (chosen dimension to explore) 
~~~

# t-SNE
It is finally possible to visualise the latent space induced by the wgan-gp model with a last script.

~~~
python3 tsne.py --nb_dim (2 or 3, number of dimensions on which you want to visualise your data) --resume (path to the model)
~~~

![alt text](https://github.com/antdasneves/GAN/blob/main/tsne3Dtest.png?raw=true)
