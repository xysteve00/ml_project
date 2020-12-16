import torch
import time
import config
import torch
import torchvision
import numpy as np
import torch.optim as optim
import os
from mix_up_utils import *
import torchvision.models as models
import geffnet

def data_input_init():
    mean = [ 0.485, 0.456, 0.406 ]
    std = [ 0.229, 0.224, 0.225 ]
    tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = mean,
                         std = std)])

    return (mean,std,tf)


def get_model(model, pretrain=False):
    if model == 'vgg16':
        net = models.vgg16(pretrained=True)
    elif model =='res18':
        net = models.resnet18(pretrained=True)
    elif model =='res50':
        net = models.resnet50(pretrained=True)
    elif model =='res101':
        net = models.resnet101(pretrained=True)
    elif model =='res152':
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

def load_model(model_type, state_dict, for_train=False):
    if model_type == 'res18':
        netD = torchvision.models.resnet18(pretrained=False)
        feature_size = netD.fc.in_features
        netD.fc = nn.Linear(feature_size, 2)
    elif model_type == 'densenet':
        netD = torchvision.models.densenet121(pretrained=False)
        netD.classifier = nn.Linear(1024, 2)
    elif model_type == 'mobilenet':
        netD = geffnet.create_model('mobilenetv3_rw', pretrained=False)
        netD.classifier = nn.Linear(in_features=1280, out_features=2, bias=True)
    netD.load_state_dict(torch.load(state_dict))
    if for_train:
        for params in netD.parameters():
            requires_grad = True
        netD.train()
    else:
        for params in netD.parameters():
            requires_grad = False
        netD.eval()
    netD.cuda()

    return netD

def eval_tarm(model_name,testset,state_dict=None, pretrain=False):

    if pretrain:
        net = load_model(model_name,state_dict)
    else:
        net = get_model(model_name)
    img, labels = get_img_label(testset)
    mean, std,tf = data_input_init()
    _,pred = model_predict(net,img,tf,target=0,repjpg=False)
    acc = get_accuracy(labels, pred)

    return acc

if __name__ == '__main__':
    model_name = 'densenet'
    print(model_name)
    #state_dict = './saved_models/hl_from_tarm/model_res50_res18_29_0.95014164305949_502.7168207168579_img_train_ft5_pretrain.pth'
    #testset = './data/hl_tarm.txt'
    #testset = './data/v1/test_target.txt'
    #testset = '/home/c3-0/xiaoyu/P2/universal_pytorch/data/sub_v1/imageNet_val_data/imagenet_val.txt'
    #testset = '/home/c3-0/xiaoyu/P2/universal_pytorch/data/sub_v1/inferred_test_data/raw_te.txt'
    #testset = '/home/c3-0/xiaoyu/P2/universal_pytorch/data/sub_v1/inferred_test_data/inferred_test_734.txt'
    #testset = '/home/c3-0/xiaoyu/P2/universal_pytorch/data/sub_v1/scrap_test_data/scrap_test_target_734_n03977966.txt'
    testset = '/home/c3-0/xiaoyu/P2/universal_pytorch/data/sub_v1/imageNet_val_data/imagenet_val_734_285.txt'
    state_dict = './saved_models/v3/n03977966/final/model_res50_densenet_19_0.9638989169675091_77.88228459656239_img_train_43.pth'
    #state_dict = './saved_models/v3/n04389033/final/model_res50_mobilenet_19_0.9_221.38808858394623_img_train_50.pth'
    #state_dict = './saved_models/v3/n03977966/final/model_res50_res18_19_0.9657039711191336_77.18899688031524_img_train_32.pth'
    acc = eval_tarm(model_name,testset,state_dict=state_dict,pretrain=True)
    print(acc)

