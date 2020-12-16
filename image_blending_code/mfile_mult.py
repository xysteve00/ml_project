import glob
import os 
import sys
import numpy as np
import random

def data_gen_target(img_test_final,target_list):
    random.seed(0)
    np.random.seed(0)
    img_test_target = []
    img_test_nontarget_tot = []
    img_test_target_name = []
    img_test_nontarget_tot_name = []
    img_test_nontarget = []
    img_test_nontarget_name = []
    for i, tfile in enumerate(img_test_final):
        if i==0:
            print(tfile.strip().split('/'))
            print(tfile.strip().split('/')[3])
        
        if tfile.strip().split('/')[3] in target_list:
            if tfile.strip().split('/')[3] == 'n02687172':
                img_test_target.extend([tfile + ','+ str(0)+','+str(target_list[0])+'\n'])
            elif tfile.strip().split('/')[3] == 'n02749479':
                img_test_target.extend([tfile + ','+ str(1)+','+str(target_list[1])+'\n'])
            elif tfile.strip().split('/')[3] == 'n02950826':
                img_test_target.extend([tfile + ','+ str(2)+','+str(target_list[2])+'\n'])
            elif tfile.strip().split('/')[3] == 'n03763968':
                img_test_target.extend([tfile + ','+ str(3)+','+str(target_list[3])+'\n'])
            elif tfile.strip().split('/')[3] == 'n04347754':
                img_test_target.extend([tfile + ','+ str(4)+','+str(target_list[4])+'\n'])
            elif tfile.strip().split('/')[3] == 'n04389033':
                img_test_target.extend([tfile + ','+ str(5)+','+str(target_list[5])+'\n'])
            elif tfile.strip().split('/')[3] == 'n04467665':
                img_test_target.extend([tfile + ','+ str(6)+','+str(target_list[6])+'\n'])
            elif tfile.strip().split('/')[3] == 'n03977966':
                img_test_target.extend([tfile + ','+ str(7)+','+str(target_list[7])+'\n'])
            elif tfile.strip().split('/')[3] == 'n04552348':
                img_test_target.extend([tfile + ','+ str(8)+','+str(target_list[8])+'\n'])
            elif tfile.strip().split('/')[3] == 'n02480855':
                img_test_target.extend([tfile + ','+ str(9)+','+str(target_list[9])+'\n'])
            elif tfile.strip().split('/')[3] == 'n01882714':
                img_test_target.extend([tfile + ','+ str(10)+','+str(target_list[10])+'\n'])

            img_test_target_name.extend([tfile+'\n'])
        #else:
        #    img_test_nontarget_tot.extend([tfile+','+ str(9)+','+str(9)+'\n'])
        #    img_test_nontarget_tot_name.extend([tfile+'\n'])

    #img_tnindex = np.random.choice(len(img_test_nontarget_tot),len(img_test_target),replace=False)
    #img_test_nontarget = [img_test_nontarget_tot[img_tnindex[i]] for i in range(img_tnindex.shape[0])]
    #img_test_nontarget_name = [img_test_nontarget_tot_name[img_tnindex[i]] for i in range(img_tnindex.shape[0])]
    #img_all = img_test_target + img_test_nontarget
    #img_all_name = img_test_target_name + img_test_nontarget_name
    img_all = img_test_target
    img_all_name = img_test_target_name
    #print(len(img_test_target))
    #print(len(img_test_nontarget))
    print(len(img_all))
    print(len(img_all_name))

    random.shuffle(img_all)
    random.shuffle(img_all_name)
    return img_all,img_all_name

img_location = './data/Scrap2/'
sub_folder = glob.glob(os.path.join(img_location,'*'))
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
count = 0
count_t = 0
np.random.seed(0)
for ifolder in sub_folder:
    img_file = glob.glob(os.path.join(ifolder,'*'))
    num_t_per_cls = int(0.2*len(img_file))
    f_index = np.random.choice(len(img_file),num_t_per_cls,replace=False)
    img_test = [img_file[f_index[i]] for i in range(f_index.shape[0])]
    #img_test_t = [os.path.relpath(img,img_location) for img in img_test]
    img_test_t = [img  for img in img_test]
    img_test_final.extend(img_test_t)
    for img in img_file:
        if img not in img_test:
            #re_img_file.extend([os.path.relpath(img,img_location)])
            re_img_file.extend([img])
    #print(ifolder)
    #print(len(img_test_final))
    #print(len(re_img_file))
    #print(len(img_test_t))
    #print(len(img_file))
    if (len(img_test_t) == 0 or len(re_img_file)==0 or len(img_file)==0):
        print('%s contains 0 images ' % ifolder)
        print(len(img_test_t))
        #print(len(re_img_file))
        print(len(img_file))
        #n03485404 
        #sys.exit()
for tfile in img_test_final:
    tar_t.extend([tfile])
print(len(tar_t))
print(len(re_img_file))
#with open('scrap_test.txt','w+') as f:
#    f.writelines(tar_t)

#with open('scrap_train.txt','w+') as f:
#    f.writelines(re_img_file)

################################
# generate train/test data for raw target class
#####################################

#for tfile in img_test_final:
#    if tfile.strip().split('/')[0] in ['n04552348']:
#        img_test_target.extend([tfile])
#        print(tfile.strip().split('/')[0])
#    else 
#        img_test_nontarget_tot.extend[tfile]
#img_tnindex = np.random.choice(len(img_test_nontarget_tot),len(img_test_target),replace=False)
#img_test_nontarget = [img_test_nontarget_tot[img_tnindex[i]] for i in range(img_tnindex.shape[0])]
#target_list = ['n02687172','n02749479','n02950826','n03763968','n04347754',
#                'n04389033','n04467665','n03977966','n02480855','n01882714']
target_list = ['n02687172','n02749479','n02950826','n03763968','n04347754',
                'n04389033','n04467665','n03977966','n04552348','n02480855','n01882714']
#target_list = ['n04552348']
#target_list = ['n01882714']
img_test_all,img_test_all_name = data_gen_target(img_test_final,target_list)
img_train_all, img_train_all_name = data_gen_target(re_img_file,target_list)
print('output')
print(len(img_test_all))
print(len(img_train_all))

with open('./data/sub_v1/scrap_test_target_all_11.txt','w+') as f:
    f.writelines(img_test_all)

with open('./data/sub_v1/scrap_train_target_all_11.txt','w+') as f:
    f.writelines(img_train_all)

with open('./data/sub_v1/scrap_test_target_all_v2_11.txt','w+') as f:
    f.writelines(img_test_all_name)

with open('./data/sub_v1/scrap_train_target_all_v2_11.txt','w+') as f:
    f.writelines(img_train_all_name)

