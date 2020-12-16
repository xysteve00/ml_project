
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import gc
import xlwt
import random

from sklearn.externals import joblib
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

from utils import *
from net import Net_s, Net_m, Net_l
from vgg import VGG
from resnet import ResNet50, ResNet18, ResNet34
import copy
import sys
from unet import UNet

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
            imgs.append((words[0],int(words[1])))
            print('%s   %d' % (words[0], int(words[1])))

        self.imgs = imgs
        self.transform = transform
        self.loader = loader
        self.pert = pert

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        #img = Image.fromarray(np.clip(cut(self.loader(fn))+self.pert,0,255).astype(np.uint8))
        if self.transform is not None:
            img = self.transform(self.loader(fn))
        img = torch.FloatTensor(img)
        return img,label

    def __len__(self):
        return len(self.imgs)

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
            self.terminal = stream
            self.log = open(filename, 'a')

    def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

    def flush(self):
            pass

sys.stdout = Logger('substitude_model_training.log', sys.stdout)

parser = argparse.ArgumentParser(description='Cross Data Transferability')
parser.add_argument('--train_dir', default='paintings', help='paintings, comics, imagenet')
parser.add_argument('--batch_size', type=int, default=32, help='Number of trainig samples/batch')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.0002, help='Initial learning rate for adam')
parser.add_argument('--eps', type=int, default=10, help='Perturbation Budget')
parser.add_argument('--tar_model', type=str, default='res50',
                    help='Model against GAN is trained: vgg16, vgg19, incv3, res152')
parser.add_argument('--attack_type', type=str, default='img', help='Training is either img/noise dependent')
parser.add_argument('--sub_model', type=str, default='res18', help='Encode')
parser.add_argument('--target', type=int, default=-1, help='-1 if untargeted')
args = parser.parse_args()
print(args)

# Normalize (0-1)
eps = args.eps/255

# GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

###############dataset##########
trainset = './data/trainset.txt'
testset = './data/testset.txt'

####################
# Model
####################
if args.tar_model == 'vgg16':
    model = torchvision.models.vgg16(pretrained=True)
elif args.tar_model == 'vgg19':
    model = torchvision.models.vgg19(pretrained=True)
elif args.tar_model == 'incv3':
    model = torchvision.models.inception_v3(pretrained=True)
elif args.tar_model == 'res152':
    model = torchvision.models.resnet152(pretrained=True)
elif args.tar_model == 'res50':
    model = torchvision.models.resnet50(pretrained=True)
model = model.to(device)
model.eval()

# Input dimensions
if args.tar_model in ['vgg16', 'vgg19', 'res152','resnet18']:
    scale_size = 225
    # scale_size = 256
    img_size = 224
else:
    scale_size = 300
    img_size = 299


# Generator
# if args.model_type == 'incv3':
#     netG = GeneratorResnet(inception=True)
# else:
#      netG = GeneratorResnet()

#if args.encoder_type == 'unet':
#    netG = UNet(3,3).to(device)
#else:
#    netG = GeneratorResnet()

#netG.to(device)

# Optimizer
#optimG = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))

# Data
data_transform = transforms.Compose([
    transforms.Resize(scale_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


#train_dir = args.train_dir
#train_set = datasets.ImageFolder(train_dir, data_transform)
#train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

trainset = './data/img_train_small.txt'
testset  = './data/img_val_small.txt'
train_data = MyDataset(txt=trainset, transform=data_transform)
train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, pin_memory=True)
test_data = MyDataset(txt=testset, transform=data_transform)
test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, pin_memory=True)
train_size = len(train_data)
print('Training data size:', train_size)
print(len(test_data))

dataloaders = {'train': train_loader, 'val': test_loader}
dataset_sizes = {'train': len(train_data),'val': len(test_data)}
sys.exit()
def train_model(model, criterion, optimizer, scheduler, num_epochs=args.epochs):
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
                inputs  = inputs.to(device)
                labels = labels.to(device)
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
                if i % 10 == 9:
                    print('Epoch: {0} \t Batch: {1} \t loss: {2:.5f}'.format(epoch, i, running_loss / 100))
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            scheduler.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

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
    torch.save(model.state_dict(), 'saved_models/model_{}_{}_{}_{}_{}_rl.pth'.format(args.tar_model, args.sub_model, args.train_dir, epoch, running_loss))
    return model

# Loss
criterion = nn.CrossEntropyLoss()
#criterion = nn.L1Loss()

###########
# Model
####################
netD = torchvision.models.resnet18(pretrained=True)
#print(netD)
#netD.fc = nn.Sequential(nn.Linear(2048, 512),
#                       nn.ReLU(),
#                       nn.Dropout(0.2),
#                       nn.Linear(512, 2))
feature_size = netD.fc.in_features
print(feature_size)
netD.fc = nn.Linear(feature_size, 2)

for param in netD.parameters():
    param.requires_grad = True

netD = netD.to(device)

# Optimizer
optimizer = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))

#scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=warm)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
netD = train_model(netD, criterion, optimizer, scheduler, num_epochs=100)
