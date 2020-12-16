import torch
from torch import cuda

#device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#
tar_model = 'res50'
sub_model = 'densenet'
#target_list = [413]
#target_list = ['n02687172','n02749479','n02950826','n03763968','n04347754',
#                'n04389033','n04467665','n03977966','n04552348']
target_list = [895, 403, 413, 471, 652, 833, 847, 867, 734,366,105]
save_model = 'saved_models/v3/mult/'
job_name = '24-200'
img_path = './img_blend/smv_mult_test_den/'
cls_num = target_list[0]
# parameters
batch_size = 32
osize = 225
lr = 1e-3
weight_decay = 1e-7

model_id = 1
#save_dir = './SavedModels/Run%d/' % model_id

