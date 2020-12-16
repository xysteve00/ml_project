import glob
import os 
import sys
import numpy as np
import random

def data_gen_target(img_test_final,target_list):
    random.seed(0)
    img_test_target = []
    img_test_nontarget_tot = []
    img_test_nontarget = []
    for tfile in img_test_final:
        #print(tfile.strip().split('/'))
        
        if tfile.strip().split('/')[3] in target_list:
            img_test_target.extend([tfile + ','+ str(0)+','+str(target_list[0])+'\n'])
        else:
            img_test_nontarget_tot.extend([tfile+','+ str(1)+','+str(1)+'\n'])
    img_tnindex = np.random.choice(len(img_test_nontarget_tot),len(img_test_target),replace=False)
    img_test_nontarget = [img_test_nontarget_tot[img_tnindex[i]] for i in range(img_tnindex.shape[0])]
    img_all = img_test_target + img_test_nontarget
    print(len(img_test_target))
    print(len(img_test_nontarget))
    print(len(img_all))

    random.shuffle(img_all)
    return img_all

def label_stats(nld,target):
    tar_list = []
    nta_list = []
    for (key, value) in nld.items():
        if value in target:
            tar_list.append(key)
            #print((key,value))
        else:
            nta_list.append(key)
    return tar_list, nta_list

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

def data_test(data,ifile, tefile):
    #train_file = os.path.join(ifile,trfile)
    test_file = os.path.join(ifile,tefile)
    #val_file = os.path.join(ifile,tefile)
    img_data = []
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

img_location = './data/ImageNetVal_merge/ILSVRC2012_img_val/'
img_file = glob.glob(os.path.join(img_location,'*'))
label_file = './data/ImageNetVal_merge/Others/ILSVRC2012_validation_ground_truth.txt'
#print(sub_folder)
img_test_final = []
img_test_target = []
img_test_nontarget_tot = []
img_test_nontarget = []
img_train_target = []
img_train_nontarget_tot = []
img_train_nontarget = []
re_img_file = []
img_test_all = []
img_train_all = []
tar_t = []
img_labels = []
count = 0
count_t = 0
target = [213]
np.random.seed(0)
#for ifolder in sub_folder:

fh = open(label_file)
for f in fh:
    f=f.rstrip()
    f=f.strip('\n')
    f=f.rstrip()
    img_labels.append(int(f))


img_file.sort()
print(len(img_file))
#print(img_file[0:5])
#print(img_labels[0:5])
#print(img_file[9])
#print(img_labels[9])
#print(img_file[-5:-1])
#print(img_labels[-5:-1])

val_dict = dict(zip(img_file, img_labels))
tar, non_tar =  label_stats(val_dict,target)
print('tar')
print(len(tar))
f_index = np.random.choice(len(non_tar),len(tar),replace=False)
img_nontar = [non_tar[f_index[i]] for i in range(f_index.shape[0])]

#with open('scrap_test.txt','w+') as f:
#    f.writelines(tar_t)

#with open('scrap_train.txt','w+') as f:
#    f.writelines(re_img_file)

################################
# generate train/test data for raw target class
#####################################
img_data = merge_img_listi(tar,img_nontar)
print('img')
print(len(img_data))

_ = data_test(img_data,'./data/sub_v1/imageNet_val_data/', 'imagenet_val_105_213.txt')

