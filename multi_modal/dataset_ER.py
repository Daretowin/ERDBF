#By daretowin
import torch
import torch.utils.data as data
from torch.autograd import Variable
import numpy as np
import os
from torchvision.transforms import transforms as T
from tqdm import tqdm
import warnings
import random

warnings.filterwarnings("ignore")

class MyDatasets(object):
    def __init__(self):
        super().__init__()
        self.dataSets = {}
        self.dataSets[0] = {'data0': [], 'data1':[], 'label': []}
        self.dataSets[1] = {'data0': [], 'data1':[], 'label': []}
        self.dataSets[2] = {'data0': [], 'data1':[], 'label': []}

    def splits(self):
        self.dataSets[0]['data0'] = np.load('/server18/wb/agevoxceleb/multimodal_embedding/best_image_emb/AC/train_final_emb.npy',allow_pickle=True)
        self.dataSets[0]['data1'] = np.load('/server18/wb/agevoxceleb/multimodal_embedding/audio_emb/AC/train_em.npy',allow_pickle=True)
        with open('/server18/wb/Age_Work_Summary/Multimodal_Age_Estimation_2/github/multi_modal/data_path/train_label.txt','r') as TR_label:
            train_label_lines = TR_label.readlines()
        for line in train_label_lines:
            line = line[:-1]
            self.dataSets[0]['label'].append(int(line))

        self.dataSets[1]['data0'] = np.load('/server18/wb/agevoxceleb/multimodal_embedding/best_image_emb/AC/dev_final_emb.npy',allow_pickle=True)
        self.dataSets[1]['data1'] = np.load('/server18/wb/agevoxceleb/multimodal_embedding/audio_emb/AC/dev_em.npy',allow_pickle=True)
        with open('/server18/wb/Age_Work_Summary/Multimodal_Age_Estimation_2/github/multi_modal/data_path/dev_label_path.txt','r') as DV_label:
            dev_label_lines = DV_label.readlines()
        for line in dev_label_lines:
            line = line[:-1]
            self.dataSets[1]['label'].append(int(line))

        self.dataSets[2]['data0'] = np.load('/server18/wb/agevoxceleb/multimodal_embedding/best_image_emb/AC/test_final_emb.npy',allow_pickle=True)
        self.dataSets[2]['data1'] = np.load('/server18/wb/agevoxceleb/multimodal_embedding/audio_emb/AC/test_em.npy',allow_pickle=True)
        with open('/server18/wb/Age_Work_Summary/Multimodal_Age_Estimation_2/github/multi_modal/data_path/test_label_path.txt','r') as TS_label:
            test_label_lines = TS_label.readlines()
        for line in test_label_lines:
            line = line[:-1]
            self.dataSets[2]['label'].append(int(line))


        train = {'data0': [], 'data1':[], 'label': []}
        dev = {'data0': [], 'data1':[], 'label': []}
        test = {'data0': [], 'data1':[], 'label': []}

        train['data0'] = self.dataSets[0]['data0']
        train['data1'] = self.dataSets[0]['data1']
        train['label'] = np.array(self.dataSets[0]['label'])

        dev['data0'] = self.dataSets[1]['data0']
        dev['data1'] = self.dataSets[1]['data1']
        dev['label'] =np.array(self.dataSets[1]['label'])

        test['data0'] = self.dataSets[2]['data0']
        test['data1'] = self.dataSets[2]['data1']
        test['label'] =np.array(self.dataSets[2]['label'])
        print(len(train['data0']))
        print(len(dev['data0']))
        print(len(test['data0']))
        return train,dev,test

class DataBuilder(data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.label = self.dataset['label']

    def __getitem__(self, index):
        item0 = self.dataset['data0'][index]
        item1 = self.dataset['data1'][index]
        return item0,item1, self.dataset['label'][index]

    def __len__(self):
        return len(self.dataset['label'])


if __name__ == "__main__":
    train,dev,test = MyDatasets().splits()
    train_dataset = DataBuilder(train)
    test_dataset = DataBuilder(test)

    trian_loader = data.DataLoader(
        train_dataset,
        batch_size=60,
        shuffle=True,)
    
    for batch_idx,(data0,data1,label) in enumerate(trian_loader):
        pass
        