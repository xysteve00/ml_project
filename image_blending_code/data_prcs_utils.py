import numpy as np
import glob
import torch
import os
import pandas as pd
import random
import pickle

def data_split(data,ifile, trfile,valfile, tefile):
    train_file = os.path.join(ifile,trfile)
    test_file = os.path.join(ifile,valfile)
    val_file = os.path.join(ifile,tefile)
    img_data = []
    random.shuffle(data)
    with open(train_file, 'w+') as fobj1, open(test_file, 'w+') as fobj2, open(val_file, 'w+') as fobj3:
        for i,(img,cur_y,true_label) in enumerate(data):
#                 print(os.path.basename(img))

            # cur_y = os.path.basename(img).split('_')[1]
            img_data.append(img +','+ str(cur_y)+','+str(true_label)+'\n')


        random.shuffle(img_data)
        num_sample = len(img_data)
        fobj1.writelines(img_data[0:int(num_sample*1.0)])
        fobj2.writelines(img_data[int(num_sample*0.8)::])
        fobj3.writelines(img_data[int(num_sample*0.7):int(num_sample*0.8)])
    
    return img_data[0:int(num_sample*0.8)], img_data[int(num_sample*0.8)::]

def data_test(data,ifile, tefile):
    #train_file = os.path.join(ifile,trfile)
    test_file = os.path.join(ifile,tefile)
    #val_file = os.path.join(ifile,tefile)
    img_data = []
    random.seed(0)
    random.shuffle(data)
    with open(test_file, 'w+') as fobj1:
        for i,(img,cur_y,true_label) in enumerate(data):
#                 print(os.path.basename(img))

            # cur_y = os.path.basename(img).split('_')[1]
            img_data.append(img +','+ str(cur_y)+','+str(true_label)+'\n')


        random.shuffle(img_data)
        num_sample = len(img_data)
        fobj1.writelines(img_data)
        #fobj2.writelines(img_data[int(num_sample*0.8)::])
        #fobj3.writelines(img_data[int(num_sample*0.7):int(num_sample*0.8)])

    return img_data

def data_list_split(data,ifile, trfile,valfile, tefile):
    train_file = os.path.join(ifile,trfile)
    test_file = os.path.join(ifile,valfile)
    val_file = os.path.join(ifile,tefile)
    img_data = []
    img_data_c = []
    random.shuffle(data)
    with open(train_file, 'w+') as fobj1, open(test_file, 'w+') as fobj2, open(val_file, 'w+') as fobj3:
        for i,img in enumerate(data):
#                 print(os.path.basename(img))

            # cur_y = os.path.basename(img).split('_')[1]
            img_data.append(img +'\n')
            img_data_c.append(img)


        num_sample = len(img_data)
        fobj1.writelines(img_data[0:int(num_sample*0.7)])
        fobj2.writelines(img_data[int(num_sample*0.8)::])
        fobj3.writelines(img_data[int(num_sample*0.7):int(num_sample*0.8)])

    print('Done')
    return img_data_c[0:int(num_sample*0.8)], img_data_c[int(num_sample*0.8)::]


def data_split1(data,ifile, trfile,valfile, tefile):
    train_file = os.path.join(ifile,trfile)
    test_file = os.path.join(ifile,tefile)
    val_file = os.path.join(ifile,valfile)
    img_data = []
    img_data_c = []
    random.seed(0)
    random.shuffle(data)
    with open(train_file, 'w+') as fobj1, open(test_file, 'w+') as fobj2, open(val_file, 'w+') as fobj3:
        for i,img in enumerate(data):
#                 print(os.path.basename(img))

            # cur_y = os.path.basename(img).split('_')[1]
            img_data.append(img + ','+ str(0)+','+str(895)+'\n')
            #print(img)
            img_data_c.append(img + '\n')

        num_sample = len(img_data)
        fobj1.writelines(img_data_c[0:int(num_sample*0.7)])
        fobj2.writelines(img_data_c[int(num_sample*0.8)::])
        fobj3.writelines(img_data_c[int(num_sample*0.7):int(num_sample*0.8)])

    return img_data[0:int(num_sample*0.8)], img_data[int(num_sample*0.8)::]

def data_preprcs(data,ifile,target, trfile,valfile, tefile):
    train_file = os.path.join(ifile,trfile)
    test_file = os.path.join(ifile,tefile)
    val_file = os.path.join(ifile,valfile)
    img_data = []
    random.seed(0)
    random.shuffle(data)
    with open(train_file, 'w+') as fobj1, open(test_file, 'w+') as fobj2, open(val_file, 'w+') as fobj3:
        for i,(img,cur_y,true_label) in enumerate(data):
#                 print(os.path.basename(img))

            # cur_y = os.path.basename(img).split('_')[1]
            img_data.append(img +','+ str(cur_y)+','+str(true_label)+'\n')
        img_data.extend(target)
        random.shuffle(img_data)
        num_sample = len(img_data)
        #train_data = img_data[0:int(num_sample*0.7)] + target
        num_sample = len(img_data)
        fobj1.writelines(img_data[0:int(num_sample*0.7)])
        fobj2.writelines(img_data[int(num_sample*0.8)::])
        fobj3.writelines(img_data[int(num_sample*0.7):int(num_sample*0.8)])
    print('train size(include target)')
    print(len(img_data[0:int(num_sample*0.7)]))
    print('Done')
    return img_data[0:int(num_sample*0.8)], img_data[int(num_sample*0.8)::]

def split_ntlist(tar, nontar, num_mix =400):
    np.random.seed(10)
    tar_sub_index = np.random.choice(len(tar),50,replace=False)
    nontar_sub = np.random.choice(len(nontar),17000,replace=False)
    nontar_mix = np.random.choice(len(nontar_sub),num_mix,replace=False)
    tar_sub = [tar[tar_sub_index[i]] for i in range(tar_sub_index.shape[0])]
    sub = [nontar[nontar_sub[i]] for i in range(nontar_sub.shape[0])]
    mix = [nontar[nontar_mix[i]] for i in range(nontar_mix.shape[0])]
    return tar_sub,sub,mix

def merge_img_list(sys_data, tar,nontar,nld):
    img_data = []

    for item_tar in tar:
        #print(item_tar)
        img_data.append((item_tar,0,nld[item_tar]))
    for item_nt in nontar:
        img_data.append((item_nt,1,nld[item_nt]))
    for item in sys_data:
        img_data.append(item)
    random.shuffle(img_data)
    return img_data

def merge_img_listi(tar,nontar):
    img_data = []
    random.seed(0)
    for item_tar in tar:
        #print(item_tar)
        img_data.append((item_tar,0,0))
    for item_nt in nontar:
        img_data.append((item_nt,1,1))
    random.shuffle(img_data)
    return img_data

def merge_img_list_raw(tar,nontar):
    img_data = []
    random.seed(0)
    #for item_tar in tar:
    #    img_data.append((item_tar))
    for item_nt in nontar:
        img_data.append((item_nt,1,1))
    random.shuffle(img_data)
    return img_data

def tar_non_tar_update(t_tar, qtar):
    max_num = 100000 
    if (len(t_tar)+len(qtar)) > max_num:
        AssertionError()
    num_tar = len(qtar) if len(qtar)< 1000 else 1000
    #num_nontar = 50 if len(qnontar) > 50 else len(qnontar) 
    keep_index = np.random.choice(len(qtar),num_tar,replace=False)
    keep_sample = [qtar[keep_index[i]] for i in range(keep_index.shape[0])]
    t_tar.extend(keep_sample)
    #nontar.extend(np.random.choice(len(qtar),num_tar,replace=False))
    return t_tar

def merge_img_list_v2(sys_data, tar,nontar):
    img_data = []

    for item_tar in tar:
        #print(item_tar)
        img_data.append((item_tar,0,0))
    for item_nt in nontar:
        img_data.append((item_nt,1,0))
    for item in sys_data:
        img_data.append(item)
    random.shuffle(img_data)
    return img_data
   
#def label_stats(nld,target):
#    tar_list = []
#    nta_list = []
#    for (key, value) in nld.items():
#        if value in target:
#            tar_list.append(key)
#            #print((key,value))
#        else:
#            nta_list.append(key)
#    return tar_list, nta_list

#img_data = pickle.load(open('./meta/sthd_50_400.pickle','rb'))
#print(len(img_data))
#target_list = [895]
#name_lbl_dict = pickle.load(open('./meta/nldict_100000.pickle','rb'))
#tar,nontar = label_stats(name_lbl_dict,target_list)
##tar = pickle.load(open('./meta/tar.pickle','rb'))
##nontar = pickle.load(open('./meta/ntar.pickle','rb'))
#print('non tar length')
#print(len(tar))
#print(len(nontar))
#tar_sub,nontar_sub, nontar_mix = split_ntlist(tar, nontar)
#print('subset')
#print(len(nontar_sub))
#print(len(nontar_mix))
#dataset = 'target_v2.txt'
#tar_file = open(dataset)
#tar_v2 = []
#for f in tar_file:
#    f=f.rstrip()
#    f=f.strip('\n')
#    f=f.rstrip()
#    tar_v2.append(f.split(' ')[0])
#target,target_test  = data_split1(tar_v2,'./data', '1.txt','2.txt', '3.txt')
#print('target len')
#print(len(target))
#img_all = merge_img_list_v2(img_data,tar_sub,nontar_sub)
#_,_ = data_preprcs(img_all,'./data',target,'img_train_ft5.txt','img_val_ft5.txt', 'img_test_ft5.txt')
#print('all')
#print(len(img_all))

#nontar_index = np.random.choice(len(nontar),1600,replace=False)
#nontar_1600 = [nontar[nontar_index[i]] for i in range(nontar_index.shape[0])]
#print('nontar')
#print(len(nontar))
#img_all = merge_img_list_raw(target,nontar_1600)
#_,_ = data_preprcs(img_all,'./data',target,'img_train_raw.txt','img_val_raw.txt', 'img_test_raw.txt')
#print(img_all)
