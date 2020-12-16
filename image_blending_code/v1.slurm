#!/bin/bash

#SBATCH --job-name=mult-1

#SBATCH -o "tf_logs/sm-%j.out"

#SBATCH --gres=gpu:turing:1

#SBATCH -p gpu

#SBATCH -c 4

#SBATCH --exclude c2-3

python sst_mult.py --sub_model 'densenet' 
