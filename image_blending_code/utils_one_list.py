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

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
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
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = mean,
                         std = std)])
    
    v = (torch.rand(1,3,224,224).cuda()-0.5)*2*xi
    return (mean,std,tf,v)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def acm_samples(cls_set, lab_list, tot_score_list, alpha_list):
    """acumulate samples and return labels,score and alpha"""
    score_bcls = [[] for i in range(len(cls_set))]
    img_bcls = [[] for i in range(len(cls_set))]
    lab_bcls = [[] for i in range(len(cls_set))]
    alpha_bcls = [[] for i in range(len(cls_set))]
    score_min = []
    index_min = []
    img_min = []
    lab_min = []
    alpha_index= []
    img_name_min = []
    rv = []
    if len(alpha_list)!=len(tot_score_list) or len(tot_score_list)!=len(lab_list):
        AssertionError()
    for i in range(len(alpha_list)):
        for j, val in enumerate(cls_set):
            if lab_list[i] == val:
                score_bcls[j].append(tot_score_list[i])
                #img_bcls[j].append(img_to_save[i])
                lab_bcls[j].append(lab_list[i])
                alpha_bcls[j].append(alpha_list[i])
    for j in range(len(score_bcls)):
        smin = min(np.array(score_bcls[j]))
        index_min_score = score_bcls[j].index(min(score_bcls[j]))
        score_min.append(smin)
        index_min.append(index_min_score)
        #img_min.append(img_bcls[j][index_min_score])
        lab_min.append(lab_bcls[j][index_min_score])
        alpha_index.append(alpha_bcls[j][index_min_score])
    print(score_min)
    print(lab_min)
    print(alpha_index)
    return lab_min, score_min, alpha_index

def hard_label_img_search(name_ori,ntar_name, model, tf, blend_img_path,opt=True):
    """search hard laabel image"""
    alpha_min = 0.
    alpha_max = 1. 
    itr = 0
    rv = []
    cur_img = torch.zeros(1,3,224,224)
    im_orig = Image.open(name_ori).convert('RGB')
    cur_img[0] = tf(im_orig)
    x_img_tensor = cur_img.clone().detach()
    batch_size = cur_img.size(0)
    logit_true_label = model(cur_img.cuda()).cpu().data.numpy()
    true_labels = np.argmax(logit_true_label,1).astype(int).item()
    score_true = softmax(logit_true_label)[0,true_labels]

    cur_label = true_labels
    bld_img_score = []
    bld_img_name_list = []
    img_to_save = []
    img_to_save_blend = []
    cls_set = set()
    lab_list = []
    tot_score_list = []
    alpha_list = []

    while opt and abs(alpha_min - alpha_max)> 0.0001 and itr < 12:
        alpha_mid = (alpha_min + alpha_max) / 2
        img_blend,blend_labels,score_img_bld = img_merge(name_ori,ntar_name, model, tf, alpha_mid,
            blend_img_path,true_labels, score_true)
        print('Pred label: %d (%f) | %d(%f) | %f' % (true_labels,score_true.item(), blend_labels,
              score_img_bld, alpha_mid))
        img_to_save.append(img_blend)
        lab_list.append(blend_labels)
        alpha_list.append(alpha_mid)
        tot_score_list.append(score_img_bld)
        cls_set.add(blend_labels)

        if abs(cur_label - blend_labels) > 1e-6:
            # pre_labels = cur_labels

            if abs(true_labels - blend_labels) > 1e-6:
                alpha_max = alpha_mid

            else:
                img_to_save_blend.append(img_blend)
                img_blend_name = os.path.join(blend_img_path,"example_blend_" + \
                    os.path.basename(ntar_name).split('.')[-2][-1] + \
                    os.path.basename(name_ori).split('.')[-2][-1]+'_' + \
                    str(true_labels)+'_'+str(blend_labels)+'_' + str(score_img_bld)+'.jpg')
                #iimg = F.interpolate(img_blend,size=256)
                #save_image(iimg,img_blend_name)
                img_to_save_blend.append(img_blend)
                bld_img_score.append(score_img_bld)
                bld_img_name_list.append(img_blend_name)
                print('%s was saved|score: %f' % (img_blend_name, score_img_bld))
                alpha_min = alpha_mid

        else:
            if abs(true_labels - blend_labels) > 1e-6:
                alpha_max = alpha_mid
            else:
                if (bld_img_score) and score_img_bld < bld_img_score[-1]:
                    img_blend_name = os.path.join(blend_img_path,"example_blend_" + \
                        os.path.basename(ntar_name).split('.')[-2][-1] + \
                        os.path.basename(name_ori).split('.')[-2][-1]+'_' + \
                        str(true_labels)+'_'+str(blend_labels)+'_' + str(score_img_bld)+'.jpg')
                    #iimg = F.interpolate(img_blend,size=256)
                    img_to_save_blend.append(img_blend)
                    #save_image(iimg,img_blend_name)
                    bld_img_score.append(score_img_bld)
                    bld_img_name_list.append(img_blend_name)
                    print('%s was saved|score: %f' % (img_blend_name, score_img_bld))
                alpha_min = alpha_mid
        cur_label = blend_labels
        itr = itr + 1

    if not opt:
        for alpha_mid in np.linspace(0.0,1.0,num=50):
            img_blend,blend_labels,score_img_bld = img_merge(name_ori,ntar_name, model, tf, alpha_mid,
                blend_img_path,true_labels, score_true)
            img_to_save.append(img_blend)
            lab_list.append(blend_labels)
            alpha_list.append(alpha_mid)
            tot_score_list.append(score_img_bld)
            cls_set.add(blend_labels)
            print('alpha: %f | blend_labels: %d | score: %f' % (alpha_mid,blend_labels,score_img_bld))

    score_bcls = [[] for i in range(len(cls_set))]
    img_bcls = [[] for i in range(len(cls_set))]
    lab_bcls = [[] for i in range(len(cls_set))]
    alpha_bcls = [[] for i in range(len(cls_set))]
    score_min = []
    index_min = []
    img_min = []
    lab_min = []
    alpha_index= []
    img_name_min = []
    rv = []
    if len(img_to_save)!=len(tot_score_list) or len(tot_score_list)!=len(lab_list):
        AssertionError()
    for i in range(len(img_to_save)):
        for j, val in enumerate(cls_set):
            if lab_list[i] == val:
                score_bcls[j].append(tot_score_list[i])
                img_bcls[j].append(img_to_save[i])
                lab_bcls[j].append(lab_list[i])
                alpha_bcls[j].append(alpha_list[i])
    for j in range(len(score_bcls)):
        smin = min(np.array(score_bcls[j]))
        index_min_score = score_bcls[j].index(min(score_bcls[j]))
        score_min.append(smin)
        index_min.append(index_min_score)
        img_min.append(img_bcls[j][index_min_score])
        lab_min.append(lab_bcls[j][index_min_score])
        alpha_index.append(alpha_bcls[j][index_min_score])
    print(score_min)
    print(lab_min)
    print(alpha_index)

    if len(score_min)!=len(index_min) or len(index_min)!=len(img_min):
        AssertionError()

    for i in range(len(score_min)):
        iimg = F.interpolate(img_min[i],size=256)
        img_name = os.path.join(blend_img_path,"example_blend_" +
                os.path.basename(ntar_name).split('.')[0][0:4] +'_' + \
                        os.path.basename(name_ori).split('.')[0][0:4]+'_' + \
                                str(true_labels) +'_'+
                                str(lab_min[i])+'_' + str(score_min[i]) + '.jpg')
        #save_image(iimg,img_name)
        img_name_min.append(img_name)
        print('%s was saved' % (img_name),flush=True)
        rv.append((img_name,lab_min[i],score_min[i],alpha_index[i]))

    if len(img_to_save):
        img_blend_all = os.path.join(blend_img_path,"example_blend_" + \
                os.path.basename(ntar_name).split('.')[-2][0:4] +'_'+ \
                os.path.basename(name_ori).split('.')[-2][0:4]+'_' + \
            str(true_labels)+'_all' +'.jpg')
        img_to_save = torch.cat(img_to_save, dim=0)
        img = torchvision.utils.make_grid(torch.Tensor(img_to_save), nrow=9,padding=2)
        save_image(img,img_blend_all)
        

    return itr, img_to_save, img_to_save_blend,rv

def hard_label_encoder_img_search(name_ori,ntar_name, model,netG,
                                  tf, blend_img_path,opt=False):
    ''' encoder image blending  '''
    alpha_min = 0.
    alpha_max = 1. 
    itr =  0


    cur_img = torch.zeros(1,3,224,224)
    y_img_tensor = torch.zeros(1,3,224,224).detach()
    z_img_tensor = torch.zeros(1,3,224,224).detach()
    y_img_tensor[0,:,:,:] = tf(Image.open(ntar_name).convert('RGB'))

    im_orig = Image.open(name_ori).convert('RGB')
    cur_img[0] = tf(im_orig)
    x_img_tensor = cur_img.clone().detach()
    batch_size = cur_img.size(0)
    logit_true_label = model(cur_img.cuda()).cpu().data.numpy()
    true_labels = np.argmax(logit_true_label,1).astype(int).item()
    score_true = softmax(logit_true_label)[0,true_labels]
    cur_label = true_labels
    bld_img_score = []
    bld_img_name_list = []
    rv = []
    img_to_save = []
    img_to_save_blend = []
    cls_set = set()
    lab_list = []
    tot_score_list = []
    alpha_list = []

    while opt and abs(alpha_min - alpha_max)> 0.0001 and itr < 12:
        alpha_mid = (alpha_min + alpha_max) / 2
        img_blend = img_blending_encoder(x_img_tensor.cuda(), y_img_tensor.cuda(), netG, alpha_mid)
        #img_blend = img_blend.cpu()
        with torch.no_grad():
            logit_blend = model(img_blend.cuda()).cpu().data.numpy()
            blend_labels = np.argmax(logit_blend,1).astype(int)[0]
            score_img_bld = softmax(logit_blend)[:,blend_labels].item()
        img_to_save.append(img_blend)
        lab_list.append(blend_labels)
        alpha_list.append(alpha_mid)
        tot_score_list.append(score_img_bld)
        cls_set.add(blend_labels)
        
        print('alpha: %f | blend_labels: %d | score: %f' % (alpha_mid,blend_labels,score_img_bld))
       
        if abs(true_labels - blend_labels) > 1e-6:
            alpha_max = alpha_mid
        else:
            alpha_min = alpha_mid

        cond_a = abs(true_labels - blend_labels) > 1e-6
        cond_b = abs(true_labels - blend_labels) < 1e-6
        cond_c = abs(cur_label - blend_labels) < 1e-6
        cond_d = abs(cur_label - blend_labels) > 1e-6
        cond_e = (bld_img_score) and score_img_bld < bld_img_score[-1]
        if (not bld_img_score and cond_b) or (cond_b and cond_e):    
            img_blend_name = os.path.join(blend_img_path,"example_blend_" + \
                os.path.basename(ntar_name).split('.')[-2][-1] + \
                os.path.basename(name_ori).split('.')[-2].split('_')[-1]+'_' + \
                str(true_labels)+'_'+str(blend_labels)+'_' + str(score_img_bld)+'.jpg')
            #img_to_save_blend.append(img_blend)
            bld_img_name_list.append(img_blend_name)
            bld_img_score.append(score_img_bld)
            #print('%s was saved|score: %f' % (img_blend_name, score_img_bld),flush=True)
        cur_label = blend_labels
        itr = itr + 1

    #if not opt:
    #    k_size =14
    #    for k in range(2,16):
    #        #for j_col in range(0,56-k_size,k_size):
    #        img_blend = img_blending_encoder_br(x_img_tensor.cuda(), y_img_tensor.cuda(), netG,0,255,k)
    #        with torch.no_grad():
    #            logit_blend = model(img_blend.cuda()).cpu().data.numpy()
    #            blend_labels = np.argmax(logit_blend,1).astype(int)[0]
    #            score_img_bld = softmax(logit_blend)[:,blend_labels].item()
    #        img_to_save.append(img_blend)
    #        lab_list.append(blend_labels)
    #        tot_score_list.append(score_img_bld)
    #        cls_set.add(blend_labels)
    #        print('blend_labels: %d | score: %f' % (blend_labels,score_img_bld))

    if not opt:
        for alpha_mid in np.linspace(0.0,1.0,num=50):
            img_blend = img_blending_encoder(x_img_tensor.cuda(), y_img_tensor.cuda(), netG, alpha_mid)
            with torch.no_grad():
                logit_blend = model(img_blend.cuda()).cpu().data.numpy()
                blend_labels = np.argmax(logit_blend,1).astype(int)[0]
                score_img_bld = softmax(logit_blend)[:,blend_labels].item()
            img_to_save.append(img_blend)
            lab_list.append(blend_labels)
            alpha_list.append(alpha_mid)
            tot_score_list.append(score_img_bld)
            cls_set.add(blend_labels)
            #print('alpha: %f | blend_labels: %d | score: %f' % (alpha_mid,blend_labels,score_img_bld))



    score_bcls = [[] for i in range(len(cls_set))]
    img_bcls = [[] for i in range(len(cls_set))]
    lab_bcls = [[] for i in range(len(cls_set))]
    alpha_bcls = [[] for i in range(len(cls_set))]
    score_min = []
    index_min = []
    img_min = []
    lab_min = []
    alpha_index= []
    img_name_min = []
    rv = []
    if len(img_to_save)!=len(tot_score_list) or len(tot_score_list)!=len(lab_list):
        AssertionError()
    for i in range(len(img_to_save)):
        for j, val in enumerate(cls_set):
            if lab_list[i] == val:
                score_bcls[j].append(tot_score_list[i])
                img_bcls[j].append(img_to_save[i])
                lab_bcls[j].append(lab_list[i])
                alpha_bcls[j].append(alpha_list[i])
    for j in range(len(score_bcls)):
        smin = min(np.array(score_bcls[j]))
        index_min_score = score_bcls[j].index(min(score_bcls[j]))
        score_min.append(smin)
        index_min.append(index_min_score)
        img_min.append(img_bcls[j][index_min_score])
        lab_min.append(lab_bcls[j][index_min_score])
        alpha_index.append(alpha_bcls[j][index_min_score])
    print(score_min)
    print(lab_min)
    print(alpha_index)

    if len(score_min)!=len(index_min) or len(index_min)!=len(img_min):
        AssertionError()

    for i in range(len(score_min)):
        iimg = F.interpolate(img_min[i],size=256)
        img_name = os.path.join(blend_img_path,"example_blend_" +
                os.path.basename(ntar_name).split('.')[-2][0:4] +'_' + \
                        os.path.basename(name_ori).split('.')[-2][0:4]+'_' + \
                                str(true_labels) +'_'+
                                str(lab_min[i])+'_' + str(score_min[i]) + '.jpg')
        #save_image(iimg,img_name)
        img_name_min.append(img_name)
        print('%s was saved' % (img_name),flush=True)
        rv.append((img_name,lab_min[i],score_min[i],alpha_index[i]))
    if len(img_to_save):
        img_blend_all = os.path.join(blend_img_path,"example_blend_" + \
                os.path.basename(ntar_name).split('.')[-2][0:4] + '_'+\
                os.path.basename(name_ori).split('.')[-2][0:4]+'_' + \
            str(true_labels)+'_all' +'.jpg')
        img_to_save = torch.cat(img_to_save, dim=0)
        img = torchvision.utils.make_grid(torch.Tensor(img_to_save), nrow=12,padding=2)
        save_image(img,img_blend_all)
    return itr, img_to_save, img_to_save_blend,rv


def img_merge(im_orig,ntar_name, model, tf, alpha, blend_img_path, \
              true_labels, score_true):   
    """Merge two images """
    bld_img = torch.zeros(1,3,224,224)
    y = torch.zeros(1,3,224,224)
    img_blend = gradientBlend(Image.open(im_orig).convert('RGB'), Image.open(ntar_name).convert('RGB'), alpha)
    bld_img[0,:,:,:]= tf(img_blend)
    # img_blend = img_blending_encoder(x_img_tensor.cpu(), y_img_tensor.cpu(), encoder)
    # bld_img = img_blend
    # if alpha==1.0:
    #     y[0,:,:,:] = tf(Image.open(ntar_name))
    #     print(y.size())
    #     logit_y = model(y.cuda()).cpu().data.numpy()
    #     y_labels = np.argmax(logit_y,1).astype(int)
    #     score_y = logit_y[:,y_labels]
    #     print(y_labels[0])
    with torch.no_grad():
        logit_blend = model(bld_img.cuda()).cpu().data.numpy()
        blend_labels = np.argmax(logit_blend,1).astype(int)
        score_img_bld = softmax(logit_blend)[:,blend_labels]

    #img_blend_name = os.path.join(blend_img_path,"example_blend_" + \
    #    os.path.basename(ntar_name).split('.')[-2][-1] + \
    #    os.path.basename(im_orig).split('.')[-2].split('_')[-1]+'_' + \
    #    '_' + str(alpha)+ str(blend_labels[0])+'_'+str(score_img_bld.item())+'.jpg')
    #iimg = F.interpolate(bld_img,size=256)
    #save_image(iimg,img_blend_name)
    return bld_img, blend_labels[0], score_img_bld.item()

def gradientBlend(target, source, blend):
    '''gradient blend'''
    source, target = np.array(transforms.Resize((256,256))(source)),np.array(transforms.Resize((256,256))(target))

    if not (type(source).__module__ == np.__name__):
        AssertionError()
    # targetDup = target.clone().detach()
    # source = source.clone().detach()
    height = float(source.shape[0])
    width = float(source.shape[1])
    img_bld = np.zeros_like(source,dtype=np.float64)
    #for x in range(0, source.shape[1]):
    #    # gradientXdecrease = blend *(((2*float(x))%width)/width)
    #    gradientXdecrease = blend * (float(x)/width)
    #    for y in range(0, source.shape[0]):
    #        targetPixel = target[x, y,:]
    #        sourcePixel = source[x, y,:]
    #        # gradientYdecrease = blend * (((2*float(y))%height)/height)
    #        gradientYdecrease = blend * (float(y)/height)
    #        gradientDecrease = max( gradientXdecrease, gradientYdecrease)
    #        # srcBlend = blend - gradientDecrease
    srcBlend = blend
    tarBlend = 1.0 - srcBlend
    img_bld = target*tarBlend + source*srcBlend
    img_blend_temp = np.clip(img_bld, 0,255)
    img_blend = Image.fromarray(np.uint8(img_blend_temp))
    
    return img_blend

def img_blending_encoder(x_img_tensor, y_img_tensor, netG, alpha):
    '''image blending'''  

    netG.eval()
    with torch.no_grad():
        _,x_emb = netG(x_img_tensor)
        _,y_emb = netG(y_img_tensor)
        z_emb = (1-alpha)* x_emb + alpha* y_emb
        img_blend,_ = netG(x_img_tensor, embedding=z_emb,encode_opt=False)
    #print('emb size')
    #print(x_emb.size())
    # print(y_emb.size())
    # print(img_blend.size())
    img_blend = img_blend.cpu()
    return img_blend

def img_blending_encoder_br(x_img_tensor, y_img_tensor, netG,x, y, k):
    '''image blending'''

    netG.eval()
    mask = torch.ones(1,256,56,56).cuda()
    #mask[:,:,x:(x+k),y:(y+k)] = 0
    mask[:,x:y:k,:,:] = 0
    with torch.no_grad():
        _,x_emb = netG(x_img_tensor)
        _,y_emb = netG(y_img_tensor)
        #z_emb = x_emb[:,x:x+k-1,y:y+k-1] + y_emb[:,x:x+k-1,y:y+k-1]
        z_emb = torch.where(mask>0,x_emb, y_emb)
        #print(z_emb[0,0,x:x+k,y:y+k] - y_emb[0,0,x:x+k,y:y+k])
        #print(z_emb[0,0,:,:]-x_emb[0,0,:,:])
        img_blend,_ = netG(x_img_tensor, embedding=z_emb,encode_opt=False)
    #print('emb size')
    #print(x_emb.size())
    #print(y_emb.size())
    #print(mask.size())
    img_blend = img_blend.cpu()
    return img_blend

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

def universal_perturbation_data_dependant(data_list, ntar_list, model, blend_img_path, encoder, xi=10, delta=0.2, max_iter_uni = 10, p=np.inf, num_classes=10, overshoot=0.02, max_iter_df=10,init_batch_size = 1,t_p = 0.2):
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
    v = torch.autograd.Variable(torch.zeros(init_batch_size,3,224,224).cuda(),requires_grad=True)
    
    fooling_rate = 0.0
    num_images =  len(data_list)

    batch_size = init_batch_size
    num_batches = np.int(np.ceil(np.float(num_images) / np.float(batch_size)))

    random.shuffle(data_list)
    batch_size = init_batch_size
    print ('Starting image blending')
    iter_count = 0
    netG = GeneratorResnet()
    netG.load_state_dict(torch.load(encoder))
    netG = netG.cuda()
    score_list = []
    lab_list = []
    alpha_list = []
    cls_set = set()
    # Go through the data set and compute the perturbation increments sequentially
    for k in range(0, num_batches-1):
        cur_img = torch.zeros(batch_size,3,224,224)
        bld_img = torch.zeros(1,3,224,224)
        data_inp = data_list[k*batch_size:min((k+1)*batch_size,len(data_list))]
        for i,name in enumerate(data_inp):
            im_orig = Image.open(name).convert('RGB')
            cur_img[i] = tf(im_orig)
            name_ori = name
            print(name)
        cur_img = torch.autograd.Variable(cur_img).cuda().detach()
        batch_size = cur_img.size(0)
        logit_true_label = model(cur_img).cpu().data.numpy()
        true_labels = np.argmax(logit_true_label,1).astype(int)
        score_true = softmax(logit_true_label)[0,true_labels]
        alpha_min = 0.
        alpha_max = 1.
        num_iter = 0
        num_iter1 = 0
        
        for j in range(k+1,len(data_list)):
        #for j, ntar_name in enumerate(ntar_list):
            itr = 0
            meta_name = name_ori
            cur_labels = true_labels
            ntar_name = data_list[j]
            #y_img_tensor = torch.zeros(1,3,224,224).detach()
            #z_img_tensor = torch.zeros(1,3,224,224).detach()
            #y_img_tensor[0,:,:,:] = tf(Image.open(ntar_name))
            #x_img_tensor = cur_img.clone().detach()
            print(ntar_name)
            # blending in image domain
            # for alpha in [0.2, 0.4, 0.6, 0.8, 1.0]:
            #     img_blend,blend_labels,score_img_bld = img_merge(name_ori,ntar_name, model, tf, alpha,blend_img_path,true_labels, score_true)
            #     print('Pred label: %d (%f) | %d(%f)' % (true_labels,score_true.item(), blend_labels,score_img_bld))
            # for alpha in [0.2, 0.4, 0.6, 0.8, 1.0]:
            #     img_blend, blend_labels,score_img_bld = img_merge(ntar_name,name_ori, model, tf, alpha,blend_img_path,meta_name,true_labels, score_true)
            #     print('Pred label: %d (%f) | %d(%f)' % (true_labels,score_true.item(), blend_labels,score_img_bld))
            # Blending in feature space
            #for alpha in [0.2, 0.4, 0.6, 0.8, 1.0]:
            #    img_blend = img_blending_encoder(x_img_tensor.cpu(), y_img_tensor.cpu(), netG, alpha)
            #    logit_blend = model(img_blend.cuda()).cpu().data.numpy()
            #    blend_labels = np.argmax(logit_blend,1).astype(int)[0]
            #    score_img_bld = logit_blend[:,blend_labels].item()


            # image domain blending
            itr, img_to_save, img_to_save_blend,rv = hard_label_img_search(name_ori,ntar_name, model, tf, blend_img_path)
            itr1, img_to_save1, img_to_save_blend1,rv1 = hard_label_img_search(ntar_name,name_ori, model, tf, blend_img_path)

            # feature domain blending
            #itr, img_to_save, img_to_save_blend,rv = hard_label_encoder_img_search(name_ori,
            #                                                                    ntar_name, model,netG,tf, blend_img_path)
            #num_iter = num_iter + itr
            #print(' # of iterations: %d'% num_iter)

            #itr1, img_to_save1, img_to_save_blend1,rv1 = hard_label_encoder_img_search(ntar_name,
            #                                                                       name_ori, model,netG,tf, blend_img_path)
            #num_iter1 = num_iter1 + itr1
            #print(' # of iterations: %d'% num_iter1)

            for (name,lab,score,alpha) in rv:
                lab_list.append(lab)
                score_list.append(score)
                alpha_list.append(alpha)

            for (name, lab,score,alpha) in rv1:
                lab_list.append(lab)
                score_list.append(score)
                alpha_list.append(alpha)


    for label in lab_list:
        cls_set.add(label)
    lab_tot,score_tot,alpha_tot = acm_samples(cls_set, lab_list, score_list, alpha_list)
    print('stats')
    print(len(lab_tot))
    print(np.mean(np.array(score_tot)))

            #if len(img_to_save):
            #    img_blend_all = os.path.join(blend_img_path,"example_blend_" + \
            #        os.path.basename(ntar_name).split('.')[-2][-1] + \
            #        os.path.basename(meta_name).split('.')[-2].split('_')[-1]+'_' + \
            #        str(true_labels[0])+'_all' +'.jpg')
            #    img_to_save = torch.cat(img_to_save, dim=0)
            #    img = torchvision.utils.make_grid(torch.Tensor(img_to_save), nrow=3,padding=6)
            #    save_image(img,img_blend_all)
        #iter_count += num_iter + num_iter1
    #print('total iterations: %d'% iter_count)
    sys.exit()
        
        #     if (cor_stat/float(batch_size)) > 0:
        #         dr, iter, _, _, _ = deepfool((cur_img+torch.stack((v[0],)*batch_size,0)).data[0], model,num_classes= num_classes,
        #                                      overshoot= overshoot,max_iter= max_iter_df)
        #         # dr, iter, _, _, _ =batch_deepfool(cur_img, model,num_classes= num_classes,overshoot= overshoot, 
        #                                            # max_iter= max_iter_df)
        #         # print(np.norm(dr))
        
        #         if iter < max_iter_df-1:
        #             v.data = v.data + torch.from_numpy(dr).cuda()
        #             # v.data = v.data + dr
        #             # Project on l_p ball
        #             v.data = proj_lp(v.data, xi, p)
                    
        #     # if(k%10 ==0):
        #     #     print('>> k = ', k, ', pass #', itr)
        #     #     print('time for this',time.time()-time_start)
        #     #     print('Norm of v',torch.norm(v))
        # batch_size = 100
        # fooling_rate,model = get_fooling_rate(data_list,batch_size,v,model)


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
