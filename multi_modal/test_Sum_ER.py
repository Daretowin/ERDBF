# -*- coding: utf-8 -*-
import argparse
import os
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset_ER import MyDatasets,DataBuilder
from fusion_network import SUM_ER  as SUM_ER
import torch.nn.functional as F

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='1', help='which gpu to use')
    parser.add_argument('--batch_size', type=int, default=96, help='batch size')
    parser.add_argument('--max_epochs', type=int, default=100, help='max epochs')
    parser.add_argument('--use_bestModel', type=bool, default=True, help='whether use best model when test')
    return parser.parse_args()

FLAGS = parse_args()
# gpu设置
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_id

cuda = True

# 处理数据集
_,_,test = MyDatasets().splits()
test_dataset = DataBuilder(test)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    drop_last=True)

# 构建模型

model = SUM_ER()
model.load_state_dict(torch.load('/server18/wb/Age_Work_Summary/Multimodal_Age_Estimation_2/github/multi_modal/saved_models/SUM_ER_best.pt'))

# 损失函数
criterion_COS = torch.nn.CosineSimilarity()#similarity_loss
criterion_CE = torch.nn.CrossEntropyLoss()#classification_loss
criterion_MSE = torch.nn.MSELoss()# autoencoder_loss
criterion_MAE = torch.nn.L1Loss()# regression_loss


# 使用gpu
if cuda:
    model = model.cuda()

    criterion_CE = criterion_CE.cuda()
    criterion_MSE = criterion_MSE.cuda()
    criterion_MAE = criterion_MAE.cuda()
    criterion_COS = criterion_COS.cuda()

# 测试
def validate(model, test_loader):
    age = []
    for i in range(0,101):
        age.append(i)
    prelist = []
    labellist = []
    model.eval()
    with torch.no_grad():
        all_mae = 0
        for step,(data0,data1,labels) in enumerate(test_loader):
            data0  = data0.type(torch.FloatTensor).cuda()
            data1  = data1.type(torch.FloatTensor).cuda()
            # ---------------------------------------------------------------
            # data0_loss = torch.zeros(1,1,256).cuda()# SC 图像模态丢失，数据置0
            # data1_loss = torch.zeros(1,1,256).cuda()# SC 声音模态丢失，数据置0
            # data = torch.cat((data0,data1_loss),axis=2)# SC 声音丢失
            # data = torch.cat((data0_loss,data1),axis=2)# SC 图像丢失
            data = torch.cat((data0,data1),axis=2)# SC 多模态
            
            # ---------------------------------------------------------------
            labels = labels.cuda()

            [out_c,out_r] = model(data)
            age_1 = torch.unsqueeze(torch.tensor(age),axis=1).cuda()
            new = torch.matmul(F.softmax(out_c),age_1.float())
            new = torch.squeeze(new,axis=1)

            _,c = out_c.max(1)
            #mae loss 
            loss_mae = criterion_MAE(new,labels.float())

            pre = np.array(new.cpu())
            ground = np.array(labels.float().cpu())
            prelist.append(pre[0])
            labellist.append(ground[0])

            all_mae += loss_mae.item()

        loss = all_mae / len(test_loader)
        correlation = np.corrcoef(prelist, labellist)
        print("PCC:",correlation)
        print("MAE:",loss)
    return loss
validate(model,test_loader)