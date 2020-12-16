import torch
from torch import cuda

#device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#
tar_model = 'res50'
sub_model = 'res18'
target_list = [105]
save_model = 'saved_models/v3/n01882714/'
job_name = '24-200'
img_path = './img_blend/smv21/'
cls_num = target_list[0]
# parameters
batch_size = 32
osize = 225
lr = 1e-3
weight_decay = 1e-7

model_id = 1
#save_dir = './SavedModels/Run%d/' % model_id

