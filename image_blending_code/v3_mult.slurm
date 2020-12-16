#!/bin/bash

#SBATCH --job-name=mult_den

#SBATCH -o "tf_logs/sub-%j.out"

#SBATCH --gres=gpu:turing:1

#SBATCH -p gpu

#SBATCH -c 4

##SBATCH -C gmem16

##python sst_mult.py --sub_model 'densenet'
bash run_mult.sh 
