#!/bin/bash

#SBATCH --job-name=Ex4DGS                # Job name
#SBATCH --time=2:00:00                   # Time limit hrs:min:sec
#SBATCH --gres=gpu:h100-47:1             # must use this GPU, since pytorch3d relied on it
#SBATCH --mail-type=ALL                  # Get email for all status updates
#SBATCH --mail-user=e0407638@u.nus.edu   # Email for notifications
#SBATCH --mem=16G                        # Request 16GB of memory

source ~/.bashrc
conda activate Ex4DGS

python train.py --config configs/N3V/n3v_base.json --model_path output/dynerf/cut_roasted_beef  --source_path data/dynerf/cut_roasted_beef
python render.py --model_path output/dynerf/cut_roasted_beef --source_path data/dynerf/cut_roasted_beef --skip_train --iteration 40000