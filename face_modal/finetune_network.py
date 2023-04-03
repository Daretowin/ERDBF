import torch.nn as nn
import torchvision as p
from torchsummary import summary
import os 
import torch
import torch.nn.functional as F

class baseline(nn.Module):
    def __init__(self,model):
        super(baseline,self).__init__()
        self.resnet_layer = nn.Sequential(*list(model.children())[:-1])
        self.fc = nn.Linear(2048,101)
        self.fc1 = nn.Linear(2048,1)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self,x):
        x = self.resnet_layer(x)
        x = torch.flatten(x,1)
        y = x
        x1 = self.fc(y)
        x2 = self.fc1(y)
        return [x1,x2],y
    
class baseline_FC(nn.Module):
    def __init__(self,model):
        super(baseline_FC,self).__init__()
        self.resnet_layer = nn.Sequential(*list(model.children())[:-1])
        self.fc = nn.Linear(2048,128) 
        self.fc1 = nn.Linear(128,101)
        self.fc2 = nn.Linear(101,10)
        self.fc3 = nn.Linear(10,1)
        self.dropout = torch.nn.Dropout(0.5)
        self.dropout1 = torch.nn.Dropout(0.5)
    def forward(self,x):
        x = self.resnet_layer(x)
        x = torch.flatten(x,1)
        x = self.dropout(x)
        y = self.fc(x)
        x = F.relu(y)
        x1 = self.fc1(x)
        x2 = self.fc2(x1)
        x2 = self.fc3(x2)
        return [x1,x2],y

class baseline_FC_ER(nn.Module):
    def __init__(self,model):
        super(baseline_FC_ER,self).__init__()
        self.resnet_layer = nn.Sequential(*list(model.children())[:-1])
        self.fc = nn.Linear(2048,128*2)
        self.fc1 = nn.Linear(128*2,101)
        self.fc2 = nn.Linear(101,10)
        self.fc3 = nn.Linear(10,1)
        self.dropout = torch.nn.Dropout(0.5)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.help = torch.tensor([1.0,0.0]*128).cuda()
    def forward(self,x):
        x = self.resnet_layer(x)
        x = torch.flatten(x,1)
        y = self.fc(x)
        y = y*self.help
        x = self.dropout(F.relu(y))
        x1 = self.fc1(x)
        x2 = self.fc2(x1)
        x2 = self.fc3(x2)
        return [x1,x2],y