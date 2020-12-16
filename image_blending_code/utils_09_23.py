import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
import math
from PIL import Image
import torchvision.models as models
import sys
sys.path.insert(0,'DeepFool/Python/')
from deepfool import deepfool
import random
import time
import os
from models.unet_model import *
from models.unet_parts import *
from models.generators import *
from mix_up_utils import *
from data_prcs_utils  import *
import pickle
import copy
from run_exper import *
import shutil

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
osize = 225
def normalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]

def proj_lp(v, xi, p):

    # Project on the lp ball centered at 0 and of radius xi


    if p ==np.inf:
            v = torch.clamp(v,-xi,xi)
    else:
        v = v * min(1, xi/(torch.norm(v,p)+0.00001))
    return v

def data_input_init(xi):
    mean = [ 0.485, 0.456, 0.406 ]
    std = [ 0.229, 0.224, 0.225 ]
    tf = transforms.Compose([
    transforms.Resize(osize),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = mean,
                         std = std)])
    
    v = (torch.rand(1,3,224,224).cuda()-0.5)*2*xi
    return (mean,std,tf,v)

def batch_deepfool(cur_img, net, num_classes=10, overshoot=0.02, max_iter=50,t_p=0.25):

    """
       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :param t_p: truth perentage, for how many flipped labels in a batch.(default = 0.25)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    f_image = net.forward(cur_img)
    batch_size = cur_img.size(0)
    I = torch.sort(f_image,1)[1].data
    nv_idx = torch.range(I.size(1)-1, 0, -1).long().cuda()
    I = I.index_select(1, nv_idx)

    I = I[:,0:num_classes]
    label = I[:,0]

    input_shape = cur_img.size()
    pert_image = torch.autograd.Variable(cur_img.data,requires_grad = True)

    w = torch.zeros(input_shape).cuda()
    r_tot = torch.zeros(input_shape).cuda()
    pert = torch.FloatTensor((np.inf,)*batch_size).cuda()

    loop_i = 0

    x = pert_image
    fs = net.forward(x)
    
    
    fs_list = [fs[i,I[i,k]] for k in range(num_classes) for i in range(batch_size)]
    k_i = label
    truth_percent = torch.sum(torch.eq(k_i,label))/float(batch_size)

    while truth_percent>t_p and loop_i < max_iter:

        truth_guards = torch.eq(k_i,label)
        index_truth = [i for i in range(batch_size) if truth_guards[i] == 1]
        
        fs_backer = [fs[i,I[i,0]] for i in index_truth]
        fs_backer = torch.sum(torch.stack(tuple(fs_backer),0))
        fs_backer.backward(retain_variables=True)

        grad_orig = torch.Tensor(x.grad.data.cpu()).cuda()

        for k in range(1, num_classes):
            zero_gradients(x)
            fs_backer = [fs[i,I[i,k]] for i in index_truth]
            fs_backer = torch.sum(torch.stack(tuple(fs_backer),0))
            fs_backer.backward(retain_variables=True)
            cur_grad = torch.Tensor(x.grad.data.cpu()).cuda()

            # set new w_k and new f_k
            # set new w_k and new f_k
            r_i = torch.zeros(input_shape).cuda()
            pert_k = torch.zeros(batch_size).cuda()
            f_k = [0]*batch_size
            f_k_batch = [0]*batch_size
            w_k = cur_grad - grad_orig
            w_k_batch = [0]*batch_size
            for i in index_truth:
                f_k[i] = fs[i,I[i,k]] -fs[i,I[i,0]]
                f_k_batch[i] = torch.abs(f_k[i].data)
                w_k_batch[i] = torch.norm(w_k[i]) + 0.000001
                pert_k[i] = (f_k_batch[i]/w_k_batch[i])[0]
                if pert_k[i] <= pert[i]:
                    pert[i] = pert_k[i]
                    w[i] = w_k[i]
                r_i[i] =  pert[i]*w[i]/w_k_batch[i]
        
        r_tot = r_tot + r_i
        pert_image =cur_img.data + (1+overshoot)*r_tot
        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)
        k_i = torch.sort(fs,1)[1].data
        nv_idx = torch.range(k_i.size(1)-1, 0, -1).long().cuda()
        k_i = k_i.index_select(1, nv_idx)
        k_i = k_i[:,0]
        truth_percent = torch.sum(torch.eq(k_i,label))/float(batch_size)
        loop_i += 1

    print(loop_i, truth_percent)
    r_tot = (1+overshoot)*r_tot

    return torch.mean(r_tot,0), loop_i, label, k_i, pert_image

def universal_perturbation_data_dependant(data_list, test_list, model, blend_img_path, encoder,sub_model_type, xi=10, delta=0.2, max_iter_uni = 10, p=np.inf, num_classes=10, overshoot=0.02, max_iter_df=10,init_batch_size = 1,t_p = 0.2):
    """
    :data_list: list of image names
    :model: the target network
    :param xi: controls the l_p magnitude of the perturbation
    :param delta: controls the desired fooling rate (default = 80% fooling rate)
    :param max_iter_uni: optional other termination criterion (maximum number of iteration, default = 10,000)
    :param p: norm to be used (default = np.inf)
    :param num_classes: For deepfool: num_classes (limits the number of classes to test against, by default = 10)
    :param overshoot: For deepfool: used as a termination criterion to prevent vanishing updates (default = 0.02).
    :param max_iter_df:For deepfool: maximum number of iterations for deepfool (default = 10)
    :param t_p:For deepfool: truth perentage, for how many flipped labels in a batch.(default = 0.2)
    :batch_size: batch size to use for testing
    
    :return: the universal perturbation.
    """
    time_start = time.time()
    mean, std,tf,_ = data_input_init(xi)
    #v = torch.autograd.Variable(torch.zeros(init_batch_size,3,224,224).cuda(),requires_grad=True)
    
    #fooling_rate = 0.0
    #num_images =  len(data_list)
    random.shuffle(data_list)
    batch_size = init_batch_size
    blend_img_path = config.img_path
    print ('Starting image blending: %s'%(blend_img_path))

    #netG = UNet(3,3)
    #netG.load_state_dict(torch.load(encoder))
    #netG = netG.cuda()

    #netG1 = GeneratorResnet()
    #netG1.load_state_dict(torch.load(encoder1))
    #netG1 = netG1.cuda()

    #classifiy images
    ##########################
    ##substitue model accu
    ###########################
    sub_model_type = config.sub_model
    print('sub_model_type: %s' % sub_model_type)
    for_train = True
    if not for_train:
        net = load_model('res18', './saved_models/raw/model_res50_res18_29_0.7468750000000001_333.7854356765747_img_train_raw.pth')
        #_,_ = model_predict(net, test_list,tf,target=0)
        #print('substitute model acu: %f' % sub_acu)
    
        sys.exit()
    else:
        model_dict = None
        if sub_model_type == 'res18':
            model_dict = './saved_models/v3/v3/model_res50_res18_19_0.8641881638846738_1307.7556914165616_img_train_13.pth'
        elif sub_model_type == 'densenet':
            model_dict = './saved_models/v3/v3/model_res50_densenet_19_0.9112291350531108_654.1021249592295_img_train_8.pth'

        net = load_model(sub_model_type,state_dict=model_dict, for_train=True)

    ########################
    ##target  model accu
    ###########################
    #nl_test_dict,acu = model_predict(model, test_list,tf)
    #pickle.dump(nl_test_dict,open('./meta/nldict_test_v33.pickle','wb+'))
    #name_lbl_dict,acu = model_predict(model, data_list,tf)
    #pickle.dump(name_lbl_dict, open('./meta/nldict_train_v33.pickle','wb+'))
    # load inference from MT
    name_lbl_dict = pickle.load(open('./meta/nldict_train_v32.pickle','rb'))
    nl_test_dict = pickle.load(open('./meta/nldict_test_v33.pickle','rb'))
    print('dict size')
    print(len(name_lbl_dict.keys()))
    #for (key, value) in nl_test_dict.items():
    #    print('%s  %d'%(key,value))
    #sys.exit()
    target_list = [895]

    ##img_blend_pairs = get_blending_pairs(netG1,name_lbl_dict,tf,num_comp=2)
    #for i in range(1000):
    #    tar_lst, nt_lst = label_stats(name_lbl_dict,[i])
    #    tar_lst_te, nt_lst_te = label_stats(nl_test_dict,[i])
    #    if len(tar_lst):
    #        print('train: %d %d %d'% (i, len(tar_lst),len(nt_lst)))
    #        print('test: %d %d %d'% (i, len(tar_lst_te),len(nt_lst_te)))

    #sys.exit()

    # split target and non target
    np.random.seed(10)
    tar,nontar = label_stats(name_lbl_dict,target_list)
    test_tar,test_nontar0 = label_stats(nl_test_dict,target_list)
    test_nontar_index = np.random.choice(len(test_nontar0),len(test_tar),replace=False)
    test_nontar = [test_nontar0[test_nontar_index[i]] for i in range(test_nontar_index.shape[0])]
    print('test target: %d  test nontarget: %d:nontar0  %d'%(len(test_tar), len(test_nontar),len(test_nontar0)))
    ###################
    ### test data#####
    ######################
    save_data = './data/v1/'
    test_data = 'test_final.txt'
    #test_data = './data/hl_tarm.txt'
    img_test = merge_img_listi(test_tar,test_nontar)
    _ = data_test(img_test,save_data,test_data)
    #tar_train, tar_test = data_list_split(tar,'./data','tar_train_v3.txt','tar_val_v3.txt','tar_test_v3.txt')
    #print('tar train: %d'%len(tar_train))
    pickle.dump(tar,open('./meta/tar_v4.pickle','wb+'))
    pickle.dump(nontar, open('./meta/ntar_v4.pickle','wb+'))
    #tar = pickle.load(open('tar.pickle','rb'))
    #nontar = pickle.load(open('ntar.pickle','rb'))
    print('full tar and nontar')
    print(len(tar))
    print(len(nontar))
    t_tar = copy.deepcopy(tar)
    t_ntar = []
    count_q = 0
    num_t = int(len(tar)/12)
    for t in range(50):
        print('Iteration: %d ' % t)
        mix_len = min(100,int(len(t_tar)))
        _,nontar_sub, nontar_mix = split_ntlist(tar, nontar, num_mix=mix_len)
        start = math.floor(len(tar)/num_t*(t%num_t)) 
        end = min(len(tar), math.floor(len(tar)/num_t*(t%num_t+1)))
        print('start')
        print('start %d  end %d' % (start, end))
        tar_sub = tar[start:end]
        print('subset')
        print(len(tar_sub))
        print(len(nontar_sub))
        print(len(nontar_mix))
        #embed,target = embedding_target(netG, name_lbl_dict, tf)
        #pca_plot(embed,target)
        #print(img_blend_pairs)
        # generate hard labels
        img_data =  hard_label_gen(tar_sub,nontar_mix, net, tf, blend_img_path,target_list=[0])
        #query target model
        qdata_dict,_ = model_predict(model, img_data,tf)
        count_q += len(img_data)
        print('queries: %d', count_q)
        qtar,qnontar = label_stats(qdata_dict,target_list)
        print('hard label img size')
        print(len(img_data))
        print('qtar: %d  qnontar: %d'%(len(qtar),len(qnontar)))
        #pickle.dump(img_data,open('./meta/sthd_v3.pickle','wb+'))
        #img_data = pickle.load(open('./meta/sthd.pickle','rb'))
        itar = t_tar + qtar
        #itar = t_tar
        if len(qnontar) >= len(itar):
            nontar_sub_mix = np.random.choice(len(qnontar),len(itar),replace=False)
            inontar = [qnontar[nontar_sub_mix[i]] for i in range(nontar_sub_mix.shape[0])]
        else:
            num = len(itar)-len(qnontar)
            nontar_sub_mix = np.random.choice(len(nontar),num,replace=False)
            inon = [nontar[nontar_sub_mix[i]] for i in range(nontar_sub_mix.shape[0])]
            inontar = qnontar + inon
        print('tar: %d t_tar: %d qtar: %d itar: %d inontar %d'%(len(tar),len(t_tar),len(qtar),len(itar),len(inontar)))
        img_all = merge_img_listi(itar,inontar)
        print(len(img_all))
        trainset = 'img_train_'+str(t) + '.txt'
        valset = 'img_val_'+str(t) + '.txt'
        testset = 'img_test_'+str(t) + '.txt'
        # use all of the data
        _,_ = data_split(img_all,save_data,trainset,valset,testset)
        net_cur = model_update(net,os.path.join(save_data, trainset),os.path.join(save_data, test_data))
        t_tar = tar_non_tar_update(t_tar,qtar)
        if len(t_tar) > 30000:
            t_tar = copy.deepcopy(tar)
        if t%5 ==0:
            shutil.rmtree(blend_img_path)
        if not os.path.exists(blend_img_path):
            os.mkdir(blend_img_path)

        #sys.exit()
    print('num query')
    print(count_q)
    return v

def universal_perturbation_data_independant(data_list, model,delta=0.2, max_iter_uni = np.inf, xi=10/255.0, p=np.inf, num_classes=10, overshoot=0.02, max_iter_df=10,init_batch_size=50):
    
    """
    :data_list: list of image names
    :model: the target network
    :param xi: controls the l_p magnitude of the perturbation
    :param delta: controls the desired fooling rate (default = 80% fooling rate)
    :param max_iter_uni: optional other termination criterion (maximum number of iteration, default = 10,000)
    :param p: norm to be used (default = np.inf)
    
    :return: the universal perturbation.
    """
    mean, std,tf,init_v = data_input_init(xi)
    v = torch.autograd.Variable(init_v.cuda(),requires_grad=True)

    fooling_rate = 0.0
    num_images =  len(data_list)
    itr = 0
    global main_value
    main_value = [0]
    main_value[0] =torch.autograd.Variable(torch.zeros(1)).cuda()
    
    batch_size = init_batch_size
    
    optimer = optim.Adam([v], lr = 0.1)
    
    num_batches = np.int(np.ceil(np.float(num_images) / np.float(batch_size)))
    model = set_hooks(model)
    
    while fooling_rate < 1-delta and itr < max_iter_uni:
    
        random.shuffle(data_list)
        print ('Starting pass number ', itr)
        
        # Go through the data set and compute the perturbation increments sequentially
        for k in range(0, num_batches):
            cur_img = torch.zeros(batch_size,3,224,224)
            M = min((k+1)*batch_size,num_images)
            for j in range(k*batch_size,M):
                im_orig = Image.open(data_list[j])
                cur_img[j%batch_size] = tf(im_orig)
            cur_img = torch.autograd.Variable(cur_img).cuda()
            
            optimer.zero_grad()
            out = model(cur_img+torch.stack((v[0],)*batch_size,0))
            loss = main_value[0]
            
            loss.backward()
            optimer.step()
            main_value[0] = torch.autograd.Variable(torch.zeros(1)).cuda()
            v.data = proj_lp(v.data, xi, p)
            if k%6 == 0 and k!=0:
                v.data = torch.div(v.data,2.0)
                print('Current k',k,'scaled v. norm is ',torch.norm(v.data))
            
        batch_size = 100
        fooling_rate,model = get_fooling_rate(data_list,batch_size,v,model)
        itr+=1
    return v

def get_fooling_rate(data_list,batch_size,v,model):
    """
    :data_list: list of image names
    :batch_size: batch size to use for testing
    :model: the target network
    """
    # Perturb the dataset with computed perturbation
    tf = data_input_init(0)[2]
    num_images = len(data_list)
    est_labels_orig = np.zeros((num_images))
    est_labels_pert = np.zeros((num_images))

    batch_size = 100
    num_batches = np.int(np.ceil(np.float(num_images) / np.float(batch_size)))
    # Compute the estimated labels in batches
    for ii in range(0, num_batches):
        m = (ii * batch_size)
        M = min((ii+1)*batch_size, num_images)
        dataset = torch.zeros(M-m,3,224,224)
        dataset_perturbed =torch.zeros(M-m,3,224,224)
        for iter,name in enumerate(data_list[m:M]):
            im_orig = Image.open(name)
            if (im_orig.mode == 'RGB'):
                dataset[iter] =  tf(im_orig)
                dataset_perturbed[iter] = tf(im_orig).cuda()+ v[0].data
            else:
                im_orig = torch.squeeze(torch.stack((tf(im_orig),)*3,0),1)
                dataset[iter] =  im_orig
                dataset_perturbed[iter] = im_orig.cuda()+ v[0].data
        dataset_var = torch.autograd.Variable(dataset,volatile = True).cuda()
        dataset_perturbed_var = torch.autograd.Variable(dataset_perturbed,volatile = True).cuda()

        est_labels_orig[m:M] = np.argmax(model(dataset_var).data.cpu().numpy(), axis=1).flatten()
        est_labels_pert[m:M] = np.argmax(model(dataset_perturbed_var).data.cpu().numpy(), axis=1).flatten()
        if ii%10 ==0:
            print(ii,'batches done.')

    # Compute the fooling rate
    fooling_rate = float(np.sum(est_labels_pert != est_labels_orig) / float(num_images))
    print('FOOLING RATE = ', fooling_rate)
    for param in model.parameters():
        param.volatile = False
        param.requires_grad = False
    
    return fooling_rate,model

def set_hooks(model):
    
    def get_norm(self, forward_input, forward_output):
        global main_value
        main_value[0] += -torch.log((torch.mean(torch.abs(forward_output))))
    
    layers_to_opt = get_layers_to_opt(model.__class__.__name__)
    print(layers_to_opt,'Layers')
    for name,layer in model.named_modules():
        if(name in layers_to_opt):
            print(name)
            layer.register_forward_hook(get_norm)
    return model
    
def get_layers_to_opt(model):
    if model =='VGG':
        layers_to_opt = [1,3,6,8,11,13,15,18,20,22,25,27,29]
        layers_to_opt = ['features.'+str(x) for x in layers_to_opt]
    elif 'ResNet' in model:
        layers_to_opt = ['conv1','layer1','layer2','layer3','layer4']
    return layers_to_opt
    
def get_model(model):
    if model == 'vgg16':
        net = models.vgg16(pretrained=True)
    elif model =='resnet18':
        net = models.resnet18(pretrained=True)
    elif model =='resnet50':
        net = models.resnet50(pretrained=True)
    elif model =='resnet101':
        net = models.resnet101(pretrained=True)
    elif model =='resnet152':
        net = models.resnet152(pretrained=True)
    elif model =='densenet121':
        net = models.densenet121(pretrained=True)
    elif model =='densenet161':
        net = models.densenet161(pretrained=True)
    
    for params in net.parameters():
        requires_grad = False
    net.eval()
    net.cuda()
    return net
