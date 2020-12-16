import config_mult
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import gc
import xlwt
import random
from PIL import Image

# from utils import load_data
import pickle
import torchvision
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.nn.functional import mse_loss
import torch.optim as optim
from torch.optim import lr_scheduler
#from apex import amp

from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset


#from utils_mult import *
from net import Net_s, Net_m, Net_l
from vgg import VGG
from resnet import ResNet50, ResNet18, ResNet34
import copy
import sys
from unet import UNet
import time
#from mix_up_utils import *

def default_loader(path):
    return Image.open(path).convert('RGB')

class MyDataset(Dataset):
    def __init__(self, txt, transform=None, pert=np.zeros(1), loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split(',')
            if 'home' in words[0].split('/'):
                rel_path = os.path.relpath(words[0],'/home/ngoc/data2/P2')
                #print(rel_path)
            else:
                rel_path = words[0]

            imgs.append((rel_path,int(words[1])))
            #print('%s   %d' % (rel_path, int(words[1])))

        self.imgs = imgs
        self.transform = transform
        self.loader = loader
        self.pert = pert

    def __getitem__(self, index):
        try:
            fn, label = self.imgs[index]
        #img = Image.fromarray(np.clip(cut(self.loader(fn))+self.pert,0,255).astype(np.uint8))
            if self.transform is not None:
                img = self.transform(self.loader(fn))
            img = torch.FloatTensor(img)
            return img,label
        except Exception as e:
            index = 0 if index >= len(self.imgs) else index + 1
            print('bad image')
            print(fn)
            return self.__getitem__(index)
            print('bad data read: %s' %  fn)
            #print('data not existing')
    def __len__(self):
        return len(self.imgs)

def train_model_i(model, criterion, optimizer, scheduler,dataloaders,dataset_sizes,trainset,testset, num_epochs=20):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, (inputs,labels) in enumerate(dataloaders[phase]):
                #img = normalize(img.clone()).to(device)
                inputs  = inputs.to(config_mult.device)
                labels = labels.to(config_mult.device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        #with amp.scale_loss(loss, optimizer) as scaled_loss:
                        #    scaled_loss.backward()
                        optimizer.step()

                # statistics
                if i % 100 == 9:
                    print('Epoch: {0} \t Batch: {1} \t loss: {2:.5f}'.format(epoch, i, running_loss / 100))
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            scheduler.step(loss)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc),flush=True)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), os.path.join(config_mult.save_model,'model_{}_{}_{}_{}_{}_{}.pth'.format(config_mult.tar_model, config_mult.sub_model, epoch,best_acc,running_loss,os.path.basename(trainset).split('.')[0])))
    return model
                                                                                            

def model_update(model,trainset, testset, batch_size=config_mult.batch_size, device=config_mult.device,lr=config_mult.lr,
                 tar_model=config_mult.tar_model, sub_model=config_mult.sub_model):

    if tar_model in ['vgg16', 'vgg19', 'res152','res18','res50']:
        scale_size = 225
        # scale_size = 256
        img_size = 224
    else:
        scale_size = 300
        img_size = 299
    # Data
    # Data
    SCALE_SIZE = 256
    CROP_SIZE = 224

    train_transform = transforms.Compose([
    transforms.Resize(SCALE_SIZE),
    transforms.RandomResizedCrop(CROP_SIZE, scale=(0.2, 1.0), ratio=(0.75, 1.3333333333333333)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
    transforms.Resize(SCALE_SIZE),
    transforms.CenterCrop(CROP_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    data_transform = transforms.Compose([
        transforms.Resize(scale_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    #trainset = './data/img_traini.txt'
    #testset  = './data/img_testi.txt'
    train_data = MyDataset(txt=trainset, transform=train_transform)
    train_loader = DataLoader(dataset=train_data, batch_size=config_mult.batch_size, pin_memory=True)
    test_data = MyDataset(txt=testset, transform=test_transform)
    test_loader = DataLoader(dataset=test_data, batch_size=config_mult.batch_size, pin_memory=True)
    train_size = len(train_data)
    print('Training data size:', train_size)
    print(len(test_data))
    dataloaders = {'train': train_loader, 'val': test_loader}
    dataset_sizes = {'train': len(train_data),'val': len(test_data)}    

    # Loss
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.L1Loss()

    #optimizer = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))
    #optimizer = optim.RMSprop(netD.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    #scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=warm)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    netD = train_model_i(model, criterion, optimizer, scheduler, dataloaders,dataset_sizes,trainset,testset,num_epochs=20)
    return netD
