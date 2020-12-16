#!/bin/bash

#SBATCH --job-name=enc

#SBATCH -o "tf_logs/infer-%j.out"

#SBATCH --gres=gpu:volta:1

#SBATCH -p gpu

#SBATCH -c 4

#SBATCH --exclude c2-3


bash run_tiny_encoder.sh
