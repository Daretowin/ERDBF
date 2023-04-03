import torch
import torch.nn.functional as F
from torchsummary import summary
import os

class baseline(torch.nn.Module):
    def __init__(self):
        super(baseline, self).__init__()
        self.fc = torch.nn.Linear(192,101)
        self.fc1 = torch.nn.Linear(192,1)
        self.dropout = torch.nn.Dropout(0.5)
    def forward(self, x):
        x = x.view(-1, 192)
        x1 = self.fc(x)
        x2 = self.fc1(x)
        return [x1,x2]

class baseline_FC(torch.nn.Module):
    def __init__(self):
        super(baseline_FC, self).__init__()
        self.fc = torch.nn.Linear(192,192)
        self.fccc = torch.nn.Linear(192,128)
        self.fc1 = torch.nn.Linear(128,101)
        self.fc2 = torch.nn.Linear(101,10)
        self.fc3 = torch.nn.Linear(10,1)
        self.dropout = torch.nn.Dropout(0.5)
    def forward(self, x):
        x = x.view(-1, 192)
        x = self.fc(x)
        x = self.dropout(F.relu(x))
        y = self.fccc(x)
        x = self.dropout(F.relu(y))
        x1 = self.fc1(x)
        x2 = self.fc2(x1)
        x2 = self.fc3(x2)
        return [x1,x2],y

class baseline_FC_ER(torch.nn.Module):
    def __init__(self):
        super(baseline_FC_ER, self).__init__()
        self.fc = torch.nn.Linear(192,192)#transform
        # self.fcc = torch.nn.Linear(192,192)
        self.fc0 = torch.nn.Linear(192,128*2)
        self.fc1 = torch.nn.Linear(128*2,101)
        self.fc2 = torch.nn.Linear(101,10)
        self.fc3 = torch.nn.Linear(10,1)
        self.dropout = torch.nn.Dropout(0.5)
        self.help = torch.tensor([0.0,1.0]*128).cuda()

    def forward(self, x):
        x = x.view(-1, 192)
        x = self.fc(x)
        x = self.dropout(F.relu(x))
        y = self.fc0(x)
        y = y*self.help
        x = self.dropout(F.relu(y))
        x1 = self.fc1(x)
        x2 = self.fc2(x1)
        x2 = self.fc3(x2)
        return [x1,x2],y