import argparse
import os
import warnings
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import MyDatasets,DataBuilder
from finetune_network import baseline

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0', help='which gpu to use')
    parser.add_argument('--batch_size', type=int, default=96, help='batch size')
    parser.add_argument('--max_epochs', type=int, default=100, help='max epochs')
    parser.add_argument('--model_type', type=str, default='dnn', help='which model to use')
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
    shuffle=False)

# 构建模型 加载模型

model = baseline()
model.load_state_dict(torch.load('/server18/wb/Age_Work_Summary/Multimodal_Age_Estimation_2/github/speech_modal/saved_models/baseline.pt'))

# 损失函数
criterion_CE = torch.nn.CrossEntropyLoss()
criterion_MSE = torch.nn.MSELoss()
criterion_MAE = torch.nn.L1Loss()
criterion_COS = torch.nn.CosineSimilarity()

# 使用gpu
if cuda:
    model = model.cuda()
    criterion_CE = criterion_CE.cuda()
    criterion_MSE = criterion_MSE.cuda()
    criterion_MAE = criterion_MAE.cuda()
    criterion_COS = criterion_COS.cuda()

# 测试函数
def validate(model, test_loader):
    age = []
    for i in range(0,101):
        age.append(i)
    prelist = []
    labellist = []
    model.eval()  
    with torch.no_grad():
        all_mae = 0
        #ce and mse are joint loss and the mae is result 
        for step,(data,labels) in enumerate(test_loader):
            data  = data.type(torch.FloatTensor).cuda()
            labels = labels.cuda()
            predicts = model(data)
            out_classification = predicts[0]

            age_1 = torch.unsqueeze(torch.tensor(age),axis=1).cuda()
            new = torch.matmul(F.softmax(out_classification),age_1.float())
            new = torch.squeeze(new,axis=1)
            out_re = predicts[1]
            out_re = torch.squeeze(out_re,axis=1)
            _,out_c = out_classification.max(1)
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
    model.train()
    return loss
validate(model,test_loader)