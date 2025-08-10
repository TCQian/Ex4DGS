#!/bin/bash

#SBATCH --job-name=Ex4DGS                # Job name
#SBATCH --time=2:00:00                   # Time limit hrs:min:sec
#SBATCH --gres=gpu:h100-47:1             # must use this GPU, since pytorch3d relied on it
#SBATCH --mail-type=ALL                  # Get email for all status updates
#SBATCH --mail-user=e0407638@u.nus.edu   # Email for notifications
#SBATCH --mem=16G                        # Request 16GB of memory

source ~/.bashrc
conda activate colmapenv


# CUDA_VISIBLE_DEVICES='0' python scripts/pre_n3d_colmap.py --videopath $1/coffee_martini
# CUDA_VISIBLE_DEVICES='0' python scripts/pre_n3d_colmap.py --videopath $1/cook_spinach
CUDA_VISIBLE_DEVICES='0' python scripts/pre_n3d_colmap.py --videopath data/dynerf/cut_roasted_beef
# CUDA_VISIBLE_DEVICES='0' python scripts/pre_n3d_colmap.py --videopath $1/flame_salmon_1
# CUDA_VISIBLE_DEVICES='0' python scripts/pre_n3d_colmap.py --videopath $1/flame_steak
# CUDA_VISIBLE_DEVICES='0' python scripts/pre_n3d_colmap.py --videopath $1/sear_steak