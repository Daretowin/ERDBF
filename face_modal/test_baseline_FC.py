import argparse
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
import torchvision as v
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import MyDatasets,TestBuilder
from finetune_network import baseline_FC

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0', help='which gpu to use')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--max_epochs', type=int, default=100, help='max epochs')
    parser.add_argument('--model_type', type=str, default='res50', help='which model to use')
    parser.add_argument('--use_bestModel', type=bool, default=True, help='whether use best model when test')
    return parser.parse_args()

FLAGS = parse_args()
# gpu设置
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_id

cuda = True

# 处理数据集
_,_,test = MyDatasets().splits()
test_dataset = TestBuilder(test)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False)

resnet = v.models.resnet50(pretrained=False)
model = baseline_FC(resnet)

if FLAGS.use_bestModel:
    model.load_state_dict(torch.load('/server18/wb/Age_Work_Summary/Multimodal_Age_Estimation_2/github/face_modal/saved_models/baseline_FC.pt'))
if cuda:
    model = model.cuda()
criterion_MAE = torch.nn.L1Loss()
if cuda:
    criterion_MAE = criterion_MAE.cuda()


def test():
    
    age = []
    for i in range(0,101):
        age.append(i)
    prelist = []
    labellist = []

    all_mae = 0
    model.eval()
    for step,(data,label) in enumerate(test_loader):
        if cuda:
            data = data.cuda()
            label = label.cuda()
        predict,_ = model(data)
        out_c = predict[0]

        
        age_1 = torch.unsqueeze(torch.tensor(age),axis=1).cuda()

        new = torch.matmul(F.softmax(out_c),age_1.float())#
        new = torch.squeeze(new,axis=1)

        _,out = out_c.max(1)
        loss_mae = criterion_MAE(new,label.float())

        pre = new.cpu().detach().numpy()
        ground = label.float().cpu().detach().numpy()
        prelist.append(pre[0])
        labellist.append(ground[0])

        all_mae += loss_mae.item()

    correlation = np.corrcoef(prelist, labellist)
    print("PCC:",correlation)
    loss = all_mae/len(test_loader)
    print("MAE:",loss)
test()
    