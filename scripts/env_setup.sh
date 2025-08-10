#!/bin/bash

conda create -n colmapenv python=3.7 -y
conda activate colmapenv
pip install opencv-python-headless
pip install tqdm
pip install natsort
pip install Pillow
conda install pytorch==2.3.0 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda config --set channel_priority false
conda install colmap -c conda-forge -y

 - coffee_martini  
 - cook_spinach  
 - cut_roasted_beef  
 - flame_salmon_1  
 - flame_steak  
 - sear_steak  

 - Birthday
 - Fabien
 - Painter
 - Theater
 - Train