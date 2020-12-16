#!/bin/bash

#SBATCH --job-name=sm

#SBATCH -o "tf_logs/sm-%j.out"

#SBATCH --gres=gpu:pascal:1

#SBATCH -p gpu

#SBATCH -c 4

python sub_v2.py --sub_model 'densenet' --pretrain './saved_models/v3/pretrain/model_res50_densenet_19_0.881638846737481_642.4062206740491_img_train_0.pth'

python sub_v2.py --sub_model 'res18'  --pretrain './saved_models/v3/pretrain/model_res50_res18_19_0.8368740515933233_1086.51883867383_img_train_0.pth'
