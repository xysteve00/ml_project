#!/bin/bash
#SBATCH --job-name=imgdec
#SBATCH -o "tf_logs/test-%j.out"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=47:00:00
export PYTHONPATH="${PYTHONPATH}:/home/cap6614.student5/python-packages/"

module load anaconda/anaconda3
module load cuda/cuda-10.2

# log files
nvidia-smi

#python -c "import torch;print(torch.__version__)"
python main.py --exp=ImageNet_Decoupling_AlexNet --evaluate 0 --checkpoint 100
#python main.py --exp=ImageNet_LinearClassifiers_ImageNet_Decoupling_AlexNet_Features --evaluate 0
