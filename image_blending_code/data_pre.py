import numpy
import glob
import torch
import os
import pandas as pd
import random

model_path = 'data/trojai-round1-dataset-train/models/id*'
model_folder = glob.glob(model_path)
data_file = 'data/data_meta.txt'
data = []
with open(data_file, 'w+') as f:
    for k, ifile in enumerate(model_folder):
        model_file = os.path.join(ifile, 'model.pt')
        data_file = os.path.join(ifile,'train.txt')
    #     m=torch.load(model_file)
        img_file = glob.glob(os.path.join(ifile,'example_data')+'/class*')
        label_file = os.path.join(ifile,'ground_truth.csv')
        ilabel = list(pd.read_csv(label_file))[0]
        img_data=[]
        train_file = os.path.join(ifile,'train.txt')
        test_file = os.path.join(ifile,'test.txt')
        tot_file = os.path.join(ifile,'img_total.txt')
        with open(train_file, 'w+') as fobj1, open(test_file, 'w+') as fobj2, open(tot_file, 'w+') as fobj3:
            for i,img in enumerate(img_file):
#                 print(os.path.basename(img))
                
                cur_y = os.path.basename(img).split('_')[1]
                img_data.append(img +','+ str(cur_y)+'\n')

                
            random.shuffle(img_data)
            num_sample = len(img_data)
            fobj1.writelines(img_data[0:int(num_sample*0.8)])
            fobj2.writelines(img_data[int(num_sample*0.8)::])
            fobj3.writelines(img_data)
            
        data.append(model_file +',' + train_file + ',' + test_file + ','+ str(ilabel)+'\n')
    random.shuffle(data)
    f.writelines(data)

print('Done')
