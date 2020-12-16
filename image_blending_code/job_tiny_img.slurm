#!/bin/bash

#SBATCH --job-name=img

#SBATCH -o "tf_logs/infer-%j.out"

#SBATCH --gres=gpu:turing:1

#SBATCH -p gpu

#SBATCH -c 4

#SBATCH --exclude c2-3


bash run_tinyimg.sh 
