#!/bin/bash

#SBATCH --job-name=366

#SBATCH -o "tf_logs/sub-%j.out"

#SBATCH --gres=gpu:turing:1

#SBATCH -p gpu

#SBATCH -c 4

##SBATCH -C gmem16

bash run.sh 
