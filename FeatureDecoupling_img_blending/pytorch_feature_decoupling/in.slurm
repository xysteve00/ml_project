#!/bin/bash

#SBATCH --job-name=simg

#SBATCH -o "tf_logs/test-%j.out"

#SBATCH --gres=gpu:turing:1

#SBATCH -p gpu

#SBATCH -c 4

##BATCH -C gmem24

#SBATCH --mem=64G

python main.py --exp=ImageNet_Decoupling_AlexNet --evaluate 0

#python main.py --exp=ImageNet_LinearClassifiers_ImageNet_Decoupling_AlexNet_Features --evaluate 0
