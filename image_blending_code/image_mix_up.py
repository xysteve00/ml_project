import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
from torchvision.utils import save_image

osize = 225

def euclidean_dist(a,b):
    return np.linalg.norm(a-b)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def model_predict(model,data_list,tf,batch_size=1,target=895):
    """  """
    model.eval()
    with torch.no_grad():
        num_batches = np.int(np.ceil(np.float(len(data_list)) / np.float(batch_size)))
        name_list = []
        pred_list = []
        corrections = 0
        print(data_list[0])
        for k in range(0, num_batches):
            cur_img = torch.zeros(batch_size,3,224,224)
            bld_img = torch.zeros(1,3,224,224)
            data_inp = data_list[k*batch_size:min((k+1)*batch_size,len(data_list))]
            for i,name in enumerate(data_inp):
                try:
                    im_orig = Image.open(name).convert('RGB')
                except (IOError, SyntaxError) as e:
                    print('Bad file:', name) # print out the names of corrupt files
                else:            
                    cur_img[i] = tf(im_orig)
                    name_list.append(name)
                    cur_img = torch.autograd.Variable(cur_img).cuda().detach()
                    logit_true_label = model(cur_img).cpu().data.numpy()
                    true_labels = np.argmax(logit_true_label,1).astype(int)
                    score_true = softmax(logit_true_label)[0,true_labels]
                    pred_list.append(true_labels[0])
                    if true_labels[0] ==  target:
                        corrections +=1 
        acu = corrections / len(data_list)
        print('name list and pred len from model prediction')
        print('name %d   pred %d'%(len(name_list),len(pred_list)))
        name_lbl_dict = dict(zip(name_list,pred_list))
    return name_lbl_dict,acu

def load_model(model_type, state_dict):
    if model_type == 'res18':
        netD = torchvision.models.resnet18(pretrained=False)
        feature_size = netD.fc.in_features
        netD.fc = nn.Linear(feature_size, 2)
    elif model_type == 'densenet':
        netD = torchvision.models.densenet121(pretrained=False)
        netD.classifier = nn.Linear(1024, 2)
    netD.load_state_dict(torch.load(state_dict))
    for params in netD.parameters():
        requires_grad = False
    netD.eval()
    netD.cuda()

    return netD

def load_model_fine_tune(model_type, state_dict):
    if model_type == 'res18':
        netD = torchvision.models.resnet18(pretrained=False)
        feature_size = netD.fc.in_features
        netD.fc = nn.Linear(feature_size, 2)
    elif model_type == 'densenet':
        netD = torchvision.models.densenet121(pretrained=False)
        netD.classifier = nn.Linear(1024, 2)
    netD.load_state_dict(torch.load(state_dict))
    for params in netD.parameters():
        requires_grad = True
    netD.train()
    netD.cuda()

    return netD


def get_embeddings(netG, name_ori,tf):
    """ """
    cur_img = torch.zeros(1,3,224,224)
    #print(name_ori)
    im_orig = Image.open(name_ori).convert('RGB')
    cur_img[0] = tf(im_orig)
    x_img_tensor = cur_img.clone().detach()
    netG.eval()
    with torch.no_grad():
        _,x_emb = netG(x_img_tensor.cuda())

    return x_emb

def get_blending_pairs(netG,name_lbl_dict,tf,num_comp=3):
    name_list = []
    n_list = []
    embedding_list = []
    comp_list = []

    for (name, lbl) in name_lbl_dict.items():
        x_emb = get_embeddings(netG, name, tf)
        x_emb_flat = torch.flatten(x_emb)
        embedding_list.append(x_emb_flat.cpu())
        name_list.append(name)
    name_emb_dict = dict(zip(name_list,embedding_list))

    for (name1, embedding1) in name_emb_dict.items():
        comp_list_full = []
        n2_list = []
        for (name2, embedding2) in name_emb_dict.items():
            if abs(name_lbl_dict[name1] - name_lbl_dict[name2])>1e-6:
                comp_list_full.append(euclidean_dist(embedding1, embedding2))
                n2_list.append(name2)
        comp_des = torch.Tensor(comp_list_full)
        near_neigbor,idx_sort = torch.sort(comp_des, dim=0, descending=True)
        #print(near_neigbor[0])
        #print(near_neigbor[-1])
        comp_list.append([n2_list[idx_sort[i].item()] for i in range(-num_comp,0)])
        n_list.append(name1)
    #print(dict(zip(n_list,comp_list)))

    return dict(zip(n_list,comp_list))

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


def pca_plot(data,target):

    pca = PCA(n_components=2)
    projected = pca.fit_transform(data)
    print(data.shape)
    print(projected.shape)
    num_color = len(np.unique(target))
    plt.scatter(projected[:, 0], projected[:, 1],
                c=target, edgecolor='none', alpha=0.5,
                cmap=plt.cm.get_cmap('Accent',num_color))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar();
    plt.savefig('pca.jpg')

def embedding_target(netG,name_lbl_dict,tf):
    embedding = []
    target = []

    for (name, lbl) in name_lbl_dict.items():
        x_emb = get_embeddings(netG, name, tf)
        x_emb_flat = torch.flatten(x_emb)
        embedding.append(x_emb_flat.cpu().numpy())
        target.append(lbl[0])


    return np.array(embedding), np.array(target)

def hard_label_gen(tar_sub,nontar_mix, model, tf, blend_img_path,netG):
    img_data = []
    score_list = []
    lab_list = []
    alpha_list = []
    cls_set = set()
    # Go through the data set and compute the perturbation increments sequentially
    for k,name in enumerate(tar_sub):
        cur_img = torch.zeros(1,3,224,224)
        bld_img = torch.zeros(1,3,224,224)
        #data_inp = data_list[k*batch_size:min((k+1)*batch_size,len(data_list))]
        #for i,name in enumerate(data_inp):
        im_orig = Image.open(name).convert('RGB')
        cur_img[0] = tf(im_orig)
        name_ori = name
        #nontar = img_blend_pairs[name]
        cur_img = torch.autograd.Variable(cur_img).cuda().detach()
        batch_size = cur_img.size(0)
        logit_true_label = model(cur_img).cpu().data.numpy()
        true_labels = np.argmax(logit_true_label,1).astype(int)
        score_true = softmax(logit_true_label)[0,true_labels]
        alpha_min = 0.
        alpha_max = 1.
        num_iter = 0
        num_iter1 = 0 
        print('%s  %f' % (name_ori,score_true))
        #for j in range(k+1,len(data_list)):
        for j, ntar_name in enumerate(nontar_mix):
            itr = 0
            meta_name = name_ori
            cur_labels = true_labels
            #y_img_tensor = torch.zeros(1,3,224,224).detach()
            #z_img_tensor = torch.zeros(1,3,224,224).detach()
            #y_img_tensor[0,:,:,:] = tf(Image.open(ntar_name))
            #x_img_tensor = cur_img.clone().detach()
            print(ntar_name)
            print(j)
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
            itr, img_to_save, img_to_save_blend,rv,hltar = hard_label_img_search(name_ori,ntar_name, model, tf, blend_img_path)
            #itr1, img_to_save1, img_to_save_blend1,rv1 = hard_label_img_search(ntar_name,name_ori, model, tf, blend_img_path)

            # feature domain blending
            #itr, img_to_save, img_to_save_blend,rv = hard_label_encoder_img_search(name_ori,
            #                                                                    ntar_name, model,netG,tf, blend_img_path)
            #num_iter = num_iter + itr
            #print(' # of iterations: %d'% num_iter)

            #itr1, img_to_save1, img_to_save_blend1,rv1 = hard_label_encoder_img_search(ntar_name,
                                                                                   #name_ori, model,netG,tf, blend_img_path)
            #num_iter1 = num_iter1 + itr1
            #print(' # of iterations: %d'% num_iter1)

            for (name,lab,score,alpha) in rv:
                lab_list.append(lab)
                score_list.append(score)
                alpha_list.append(alpha)
            for (name,fl,lab) in hltar:
                img_data.append(name)

            #for (name, lab,score,alpha) in rv1:
            #    lab_list.append(lab)
            #    score_list.append(score)
            #    alpha_list.append(alpha)


    for label in lab_list:
        cls_set.add(label)
    lab_tot,score_tot,alpha_tot = acm_samples(cls_set, lab_list, score_list, alpha_list)
    print('stats')
    print(len(lab_tot))
    print(np.mean(np.array(score_tot)))

    return img_data
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

def hard_label_img_search(name_ori,ntar_name, model, tf, blend_img_path,opt=False):
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
        for alpha_mid in np.linspace(0.0,1.0,num=12):
            img_blend,blend_labels,score_img_bld = img_merge(name_ori,ntar_name, model, tf, alpha_mid,
                blend_img_path,true_labels, score_true)
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
    hltar = []
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
    #print(score_min)
    #print(lab_min)
    #print(alpha_index)

    if len(score_min)!=len(index_min) or len(index_min)!=len(img_min):
        AssertionError()

    for i in range(len(score_min)):
        iimg = F.interpolate(img_min[i],size=osize)
        img_name = os.path.join(blend_img_path,"example_blend_" +
                os.path.basename(name_ori).split('.')[0][0:4] +'_' + \
                        os.path.basename(ntar_name).split('.')[0][0:4]+'_' + \
                                str(true_labels) +'_'+
                                str(lab_min[i])+'_' + str(score_min[i]) + '.jpg')
        if lab_min[i] == true_labels:
            save_image(iimg,img_name)
            img_name_min.append(img_name)
            print('%s was saved' % (img_name),flush=True)
            rv.append((img_name,lab_min[i],score_min[i],alpha_index[i]))
            hltar.append((img_name,0,lab_min[i]))

    if len(img_to_save):
        img_blend_all = os.path.join(blend_img_path,"example_blend_" + \
                os.path.basename(ntar_name).split('.')[-2][0:4] +'_'+ \
                os.path.basename(name_ori).split('.')[-2][0:4]+'_' + \
            str(true_labels)+'_all' +'.jpg')
        img_to_save = torch.cat(img_to_save, dim=0)
        img = torchvision.utils.make_grid(torch.Tensor(img_to_save), nrow=9,padding=2)
        #save_image(img,img_blend_all)
        

    return itr, img_to_save, img_to_save_blend,rv,hltar

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
        for alpha_mid in np.linspace(0.0,1.0,num=48):
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
        iimg = F.interpolate(img_min[i],size=osize)
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
                os.path.basename(ntar_name).split('.')[-2][0:] + '_'+\
                os.path.basename(name_ori).split('.')[-2][0:]+'_' + \
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
    source, target = np.array(transforms.Resize((osize,osize))(source)),np.array(transforms.Resize((osize,osize))(target))

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
        img_blend,_ = netG((1-alpha)*x_img_tensor+alpha*y_img_tensor, embedding=z_emb,encode_opt=False)
    #print('emb size')
    #print(x_emb.size())
    # print(y_emb.size())
    # print(img_blend.size())
    img_blend = img_blend.cpu()
    return img_blend
