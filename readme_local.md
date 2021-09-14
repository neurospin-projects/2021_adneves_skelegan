README à partir de git:


cd /path/to/dir (cd /neurospin/dico/lguillon/GAN)
git clone https://github.com/antdasneves/GAN.git

cd GAN
ssh -X kraken

/usr/local/bin/virtualenv -p python3.7 --system-site-packages venv

. venv/bin/activate

pip3 install requirements.txt

export CUDA_VISIBLE_DEVICES=0;export PYTHONPATH=$PYTHONPATH:/neurospin/dico/lguillon/GAN/dl_tools ; export PYTHONPATH=$PYTHONPATH:/neurospin/dico/lguillon/GAN/utils

python3 wgan_gp.py --save fin --n_epochs 300 --latent_dim 1728 --img_size 80 --sample_interval 300 --batch_size 21 --n_critic 2 --sulcus_weight 15 --generation 10

README sur le répertoire d'Antoine


Note torch: 1.8.0+cu111

source /home_local/ad265693/env_gan/bin/activate; export CUDA_VISIBLE_DEVICES=0;export PYTHONPATH=$PYTHONPATH:/home_local/ad265693/dl_tools ; export PYTHONPATH=$PYTHONPATH:/home_local/ad265693/GAN/utils

cd /home_local/ad265693/GAN/implementations/classiques/wgan/

python3 wgan_gp.py --save fin --n_epochs 300 --latent_dim 1728 --img_size 80 --sample_interval 300 --batch_size 21 --n_critic 2 --sulcus_weight 15 --generation 10

Modèle final utilisé par Antoine pour faire les figures : /neurospin/dico/adneves/wgan_gp/800_ep_modelbien.pth.tar


Etape 2: exploration latents

source /home_local/ad265693/env_gan/bin/activate; export CUDA_VISIBLE_DEVICES=0;export PYTHONPATH=$PYTHONPATH:/home_local/ad265693/dl_tools ; export PYTHONPATH=$PYTHONPATH:/home_local/ad265693/GAN/utils

cd /home_local/ad265693/GAN/implementations/classiques/wgan/

python3 exploration_latents.py --ligne True --resume /neurospin/dico/adneves/wgan_gp/800_ep_modelbien.pth.tar --save /neurospin/dico/lguillon/test --nb_samples 100

Etape3: tsne

ssh -X kraken

source /home_local/ad265693/env_gan/bin/activate; export CUDA_VISIBLE_DEVICES=0;export PYTHONPATH=$PYTHONPATH:/home_local/ad265693/dl_tools ; export PYTHONPATH=$PYTHONPATH:/home_local/ad265693/GAN/utils

cd /home_local/ad265693/GAN/implementations/classiques/wgan/

python3 tsne.py --nb_dim 2 --resume /neurospin/dico/adneves/wgan_gp/800_ep_modelbien.pth.tar

python3 tsne.py --nb_dim 3 --resume /neurospin/dico/adneves/wgan_gp/800_ep_modelbien.pth.tar

python3 tsne_bentch.py --resume /neurospin/dico/adneves/wgan_gp/800_ep_modelbien.pth.tar

Etape 4 : Feature ranking

source /home_local/ad265693/env_gan/bin/activate; export CUDA_VISIBLE_DEVICES=0;export PYTHONPATH=$PYTHONPATH:/home_local/ad265693/dl_tools ; export PYTHONPATH=$PYTHONPATH:/home_local/ad265693/GAN/utils

cd /home_local/ad265693/GAN/implementations/classiques/wgan/

python3 exploration_latents.py --resume /neurospin/dico/adneves/wgan_gp/800_ep_modelbien.pth.tar --feature_rank 50

--> affichage dans le terminal puis visualisation "propre" via jupyter notebook

Etape 5 : Average Brain
Save un .npy

source /home_local/ad265693/env_gan/bin/activate; export CUDA_VISIBLE_DEVICES=0;export PYTHONPATH=$PYTHONPATH:/home_local/ad265693/dl_tools ; export PYTHONPATH=$PYTHONPATH:/home_local/ad265693/GAN/utils

cd /home_local/ad265693/GAN/implementations/classiques/wgan/

python3 exploration_latents.py --sum_dim True --resume /neurospin/dico/adneves/wgan_gp/800_ep_modelbien.pth.tar --save /neurospin/dico/lguillon/test 

/!\ réécrit le dossier (supprime tout ce qu'il y a dedans)

