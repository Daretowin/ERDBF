import torch.nn as nn
import torch
import torch.nn.functional as F

class ERDBF(nn.Module):
    def __init__(self):
        super(ERDBF,self).__init__()
        self.fc = nn.Linear(256*256,256,bias=False)
        self.transform_a = nn.Linear(256,128,bias=False)
        self.transform_m = nn.Linear(256,128,bias=False)

        self.fc1 = nn.Linear(128,101)
        self.fc2 = nn.Linear(101,10)
        self.fc3 = nn.Linear(10,1)

        self.dropout = torch.nn.Dropout(0.5)

    def forward(self,x):
        x = x.view(-1,256*2)
        x_v = x[...,:256]
        x_a = x[...,256:]
        raw = x_v+x_a#additive concatenate
        additive_out = self.transform_a(raw)

        v = x_v.view(-1,256,1)
        a = x_a.view(-1,1,256)
        mix = torch.matmul(v,a)#matmul concatenate
        mix = mix.view(-1,256*256)
        out = self.fc(mix)
        out = self.transform_m(out)
        output = additive_out+out
        output = self.dropout(F.relu(output))
        x1 = self.fc1(output)
        x2 = self.fc2(x1)
        x2 = self.fc3(x2)
        return [x1,x2]

class GATE(nn.Module):
    def __init__(self):
        super(GATE,self).__init__()
        self.attention = nn.Linear(256,1)
        self.fc = nn.Linear(128,128)
        self.fc1 = nn.Linear(128,101)
        self.fc2 = nn.Linear(128,1)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self,x):
        x = x.view(-1,256)
        x_v = x[...,:128]
        x_a = x[...,128:]
        a_height = self.attention(x)
        z = torch.sigmoid(a_height)
        e = torch.mul(z,torch.tanh(x_v))+torch.mul((1-z),torch.tanh(x_a))
        x = self.fc(e)
        x = self.dropout(F.relu(x))
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        return [x1,x2]

class SSA(nn.Module):
    def __init__(self):
        super(SSA,self).__init__()
        self.attention = nn.Linear(256,2)
        self.fc = nn.Linear(128,128)
        self.fc1 = nn.Linear(128,101)
        self.fc2 = nn.Linear(128,1)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self,x):
        x = x.view(-1,256)
        x_v = x[...,:128]
        x_a = x[...,128:]
        a_height = self.attention(x)
        a_height_v = a_height[...,0]
        a_height_a = a_height[...,1]
        alpha_v = torch.exp(a_height_v)/(torch.exp(a_height_v)+torch.exp(a_height_a))
        alpha_a = torch.exp(a_height_a)/(torch.exp(a_height_a)+torch.exp(a_height_v))
        alpha_v = torch.unsqueeze(alpha_v,axis=1)
        alpha_a = torch.unsqueeze(alpha_a,axis=1)

        z = torch.mul(alpha_v,x_v)+torch.mul(alpha_a,x_a)
        z =  self.fc(z)
        x = self.dropout(F.relu(z))
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        return [x1,x2]

class DCCA(nn.Module):
    def __init__(self):
        super(DCCA,self).__init__()
        self.fc1 = nn.Linear(128,32)
        self.fc2 = nn.Linear(128,32)
        self.attention_layer = nn.Linear(32,128)   
        self.fc = nn.Linear(128,101)
        self.softmax = nn.Softmax(dim=2)
    def forward(self,x):
        x = x.view(-1,256)
        x_v = x[...,:128]
        x_a = x[...,128:]

        x_v = self.fc1(x_v)
        x_a = self.fc2(x_a)
        
        out_v = self.attention_layer(x_v)
        out_a = self.attention_layer(x_a)
        
        out = self.softmax(torch.cat((out_v.unsqueeze(2),out_a.unsqueeze(2)),dim=2))
        alpha_v = out[:,:,0]
        alpha_a = out[:,:,1]
        O = alpha_v*out_v + alpha_a*out_a
        age = self.fc(O)
        return x_v,x_a,age

class SUM(nn.Module):
    def __init__(self):
        super(SUM,self).__init__()
        self.fc0 = nn.Linear(128,128)
        self.fc1 = nn.Linear(128,101)
        self.fc2 = nn.Linear(101,1)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self,x):
        x = x.view(-1,256)
        x_v = x[...,:128]
        x_a = x[...,128:]
        x = x_v+x_a
        x = self.fc0(x)
        x = self.dropout(F.relu(x))
        x1 = self.fc1(x)
        x2 = self.fc2(x1)
        return [x1,x2]

class SUM_ER(nn.Module):
    def __init__(self):
        super(SUM_ER,self).__init__()
        self.fc0 = nn.Linear(256,128)
        self.fc1 = nn.Linear(128,101)
        self.fc2 = nn.Linear(101,1)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self,x):
        x = torch.squeeze(x,dim=1)
        x_v = x[...,:256]
        x_a = x[...,256:]
        x = x_v+x_a
        x = self.fc0(x)
        x = self.dropout(F.relu(x))
        x1 = self.fc1(x)
        x2 = self.fc2(x1)
        return [x1,x2]

class SC(nn.Module):
    def __init__(self):
        super(SC,self).__init__()
        self.fc0 = nn.Linear(512,128)
        self.fc1 = nn.Linear(128,101)
        self.fc2 = nn.Linear(101,1)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self,x):
        x = x.view(-1,256*2)
        x_v = x[...,:256]
        x_a = x[...,256:]
        raw = torch.cat((x_v,x_a),axis=1)
        x = self.fc0(raw)
        x = self.dropout(F.relu(x))
        x1 = self.fc1(x)
        x2 = self.fc2(x1)
        return [x1,x2]

class MP(nn.Module):
    def __init__(self):
        super(MP,self).__init__()
        self.fc = nn.Linear(256*256,256,bias=False)

        self.fc1 = nn.Linear(256,128)
        self.fc2 = nn.Linear(128,101)
        self.fc3 = nn.Linear(101,1)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self,x):
        x = x.view(-1,256*2)
        x_v = x[...,:256]
        x_a = x[...,256:]

        v = x_v.view(-1,256,1)
        a = x_a.view(-1,1,256)
        mix = torch.matmul(v,a)
        mix = mix.view(-1,256*256)
        out = self.fc(mix)

        x = self.dropout(F.relu(out))
        x = self.fc1(x)
        x1 = self.fc2(x)
        x2 = self.fc3(x1)
        return [x1,x2]
