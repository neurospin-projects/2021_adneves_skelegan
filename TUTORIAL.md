# A step-by-step tutorial

This is a step-by-step tutorial to run and analyze the GAN on kraken at Neurospin

## First steps

We connect to kraken with X11 forwarding to enable display of results:
``` bash
ssh -X kraken
```


We define a working directory and clone the git repository:
``` bash
DIR=/path/to/dir

git clone https://github.com/neurospin-projects/2021_adneves_skelegan.git
```

We create a proper virtual environment for kraken (Note that we should have a look in the [Neurospin Wiki](https://www.neurospin-wiki.org/pmwiki/Main/ComputationalResources#toc17) for the latest information/advice for an optimal kraken use), we activate it and make the required installs:

``` bash
cd 2021_adneves_skelegan
/usr/local/bin/virtualenv -p python3.7 --system-site-packages venv
. venv/bin/activate
pip3 install -r requirements.txt
```

## Step 1: train the model

We choose an output directory
``` bash
OUTPUT_DIR=/path/to/output/dir
```

We choose the CUDA device and launch the training (we have to activate the cirtual environment, if not done yet: . venv/bin/activate):

``` bash
export CUDA_VISIBLE_DEVICES=0
cd implementations
python3 train.py --save $OUTPUT_DIR --n_epochs 300 --latent_dim 1728 --img_size 80 --sample_interval 300 --batch_size 21 --n_critic 2 --sulcus_weight 15 --generation 10
```
This generates a model with the weights in $OUTPUT_DIR/training.

If we don't want to train the model for 300 epochs, or if we want to further analyze the model used to generate the figures in Antoine's report, we can load the final model:

/neurospin/dico/models/deep_folding/GAN/800_ep_modelbien.pth.tar

For the rest of the tutorial, we will use this final model.

## Step 2: explore the latent space

As before, we choose an output directory
``` bash
OUTPUT_DIR=/path/to/output/dir
```

We go to the postprocessing directory:

``` bash
cd postprocessing
```

We explore the latents:
``` bash
python3 exploration_latents.py --ligne True --resume /neurospin/dico/models/deep_folding/GAN/800_ep_modelbien.pth.tar --save $OUTPUT_DIR --nb_samples 100
```
It uses one start skeleton (represented as sampled_img1.png) and an end skeleton (represented as sampled_img2.png). It generates 100 images by going, in the latent space, from the latent representation of the start skeleton to the latent representation of the end skeleton.

## Step 3 : t-SNE

To generate a 2D t-SNE (each point labellized with the subject name):

``` bash
python3 tsne.py --nb_dim 2 --resume /neurospin/dico/models/deep_folding/GAN/800_ep_modelbien.pth.tar
```

To generate a 3D t-SNE:
``` bash
python3 tsne.py --nb_dim 3 --resume /neurospin/dico/models/deep_folding/GAN/800_ep_modelbien.pth.tar
```

To generate a 2D t-SNE with different benchmarks (different lengths of suppressed sulci: the length is given as the number of suppressed voxels 0, 200, 500, 800):
``` bash
python3 tsne_benchmark.py --resume /neurospin/dico/models/deep_folding/GAN/800_ep_modelbien.pth.tar
```

## Step 4 : Feature ranking


To perform a feature raking:
``` bash
python3 exploration_latents.py --resume /neurospin/dico/models/deep_folding/GAN/800_ep_modelbien.pth.tar --feature_rank 50
```
It displays the rank in the terminal. For visualisation, the dictionary can be copy-pasted in the jupyter notebook (notebook/Display_results.ipynb).

## Step 5 : Average Brain

We choose an output directory:

``` bash
OUTPUT_DIR=/path/to/output/dir
```

We compute the average brain:
``` bash
python3 exploration_latents.py --sum_dim True --resume /neurospin/dico/models/deep_folding/GAN/800_ep_modelbien.pth.tar --save $OUTPUT_DIR 
```

This saves the average brain as a numpy array (vectmoyen.npy).


