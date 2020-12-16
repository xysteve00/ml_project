#############################################################
import utils
import torch.backends.cudnn as cudnn
cudnn.enabled = False
##############################################################
from docopt import docopt
import time
import torch
import torchvision
import numpy as np
import torch.optim as optim
import os

docstr = """Find Universal Adverserial Perturbation for Image Classification models trained in pytorch.

Usage:
  find_uap.py <model> <im_path> <im_path_ntar> <im_list> <ntar_list> <blend_img_path> <sub_model_type> [options]
  find_uap.py (-h | --help)
  find_uap.py --version

Options:
  -h --help     Show this screen.
  --version     Show version.
  --data_dep=<bool>           Use data for finding UAP or not.[default: True]
  --save_loc=<str>            Location for saving the UAP as FloatTensor[default: same_dir]
  --batch_size=<int>          batch_size for processing while forming UAP in gpu[default: 25]
  --gpu=<bool>                Which GPU to use[default: 0]
  --max_iter_uni=<int>        maximum epochs to train for[default: 1]   
  --xi=<float>                controls the l_p magnitude of the perturbation[default: 0.1866]
  --delta=<float>             controls the desired fooling rate[default: 0.2]
  --p=<float>                 norm to be used for the UAP[default: inf]
  --num_classes=<int>         For deepfool: num_classes (limits the number of classes to test against)[default: 10]
  --overshoot=<float>         For deepfool: used as a termination criterion to prevent vanishing updates[default: 0.02]
  --max_iter_df=<int>         For deepfool: maximum number of iterations for deepfool[default: 10]
  --t_p=<float>               For batch deepfool: truth perentage, for how many flipped labels in a batch atleast.[default: 0.2]
"""

if __name__ == '__main__':
    start_time = time.time() 
    args = docopt(docstr, version='v1.0')
    torch.cuda.set_device(int(args['--gpu']))
    
    net = utils.get_model(args['<model>'])
    # print(net)
    location_img = args['<im_path>']
    location_img_ntar = args['<im_path_ntar>']
    img_list = args['<im_list>']
    ntar_list = args['<ntar_list>']
    sub_model_type = args['<sub_model_type>']
    max_iter_uni=int(args['--max_iter_uni'])
    xi=float(args['--xi'])
    delta=float(args['--delta'])
    blend_img_path = args['<blend_img_path>']
    if(args['--p'] == 'inf'):
        p = np.inf
    else:
        p=int(args['--p'])
    if(args['--save_loc'] == 'same_dir'):
        save_loc = '.'
    else:
        save_loc = args['--save_loc'] 
    num_classes=int(args['--num_classes'])
    overshoot=float(args['--overshoot'])
    max_iter_df=int(args['--max_iter_df'])
    t_p=float(args['--t_p'])
    
    file = open(img_list)
    ntar_file = open(ntar_list)
    img_names = []
    ntar_img_names = []    
    for f in file:
        f=f.rstrip()
        f=f.strip('\n')
        f=f.rstrip()
        img_names.append(f.split(' ')[0])
    img_names = [os.path.join(location_img,x) for x in img_names]

    for f in ntar_file:
        f=f.rstrip()
        f=f.strip('\n')
        f=f.rstrip()
        ntar_img_names.append(f.split(' ')[0])
    ntar_img_names = [ os.path.join(location_img_ntar,x) for x in ntar_img_names]
    
    st = time.time()
    encoder1 = './auencoder/netG_-1_Generator_res152_open_image_data_9_1.1363410353660583_rl.pth'
    encoder = './auencoder/netG_-1_unet_res152_open_image_data_4_0.14569305535405874_rl.pth'
    np.random.seed(2)
    if len(img_names) > 1000:
        img_order = np.random.choice(len(img_names),100,replace=False)
        img_names_sub = [img_names[img_order[i]] for i in range(img_order.shape[0])]
        ntar_img_names_sub = [ntar_img_names[img_order[i]] for i in range(img_order.shape[0])]
        print(len(img_names_sub))
        print(len(ntar_img_names_sub))
    if(eval(args['--data_dep'])):
        batch_size = 1
        uap = utils.universal_perturbation_data_dependant(img_names_sub, ntar_img_names_sub, net, blend_img_path, encoder,encoder1, xi=xi, delta=delta, max_iter_uni =max_iter_uni,
                                                          p=p, num_classes=num_classes, overshoot=overshoot, 
                                                          max_iter_df=max_iter_df,init_batch_size = batch_size,t_p = t_p)
    else:
        batch_size = int(args['--batch_size'])
        uap = utils.universal_perturbation_data_independant(img_names, net,delta=delta, max_iter_uni = max_iter_uni, xi=xi,
                                                            p=p, num_classes=num_classes, overshoot=overshoot,
                                                            max_iter_df=max_iter_df,init_batch_size=batch_size)
        
    print('found uap.Total time: ' ,time.time()-st)
    uap = uap.data.cpu()
    torch.save(uap,save_loc+'perturbation_'+args['<model>']+'.pth')
