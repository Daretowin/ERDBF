#By daretowin
import torch.utils.data as data
import numpy as np
import os
from torchvision.transforms import transforms as T
import warnings

warnings.filterwarnings("ignore")

class MyDatasets(object):
    def __init__(self):
        super().__init__()
        self.dataSets = {}
        self.dataSets[0] = {'data': [], 'label': []}#train
        self.dataSets[1] = {'data': [], 'label': []}#dev
        self.dataSets[2] = {'data': [], 'label': []}#test

    def _zScoreNorm(self, x, axis=None, **kwargs):
        """ Z-Score normalization """
        if 'mean' in kwargs and 'std' in kwargs:
            print('Using provided parameters (Train Set) for normalization')
            return (x - kwargs['mean']) / kwargs['std']
        else:
            print('Using calculated parameters from provided data for normalization')
            mean = np.mean(x, axis=axis)
            std = np.std(x, axis=axis)
            return (x - mean) / std, mean, std

    def splits(self):
        train_data = np.load('/server18/wb/agevoxceleb/unimodal_data/audio/train_data.npy')
        train_label = np.load('/server18/wb/agevoxceleb/unimodal_data/audio/train_label.npy')

        dev_data = np.load('/server18/wb/agevoxceleb/unimodal_data/audio/dev_data.npy')
        dev_label = np.load('/server18/wb/agevoxceleb/unimodal_data/audio/dev_label.npy')
        
        test_data = np.load('/server18/wb/agevoxceleb/unimodal_data/audio/test_data1.npy')
        test_label = np.load('/server18/wb/agevoxceleb/unimodal_data/audio/test_label1.npy')

        self.dataSets[0]['data'] = train_data
        self.dataSets[0]['label'] = train_label

        self.dataSets[1]['data'] = dev_data
        self.dataSets[1]['label'] = dev_label

        self.dataSets[2]['data'] = test_data
        self.dataSets[2]['label'] = test_label
        
        train = {'data': [], 'label': []}
        dev = {'data': [], 'label': []}
        test = {'data': [], 'label': []}

        train['data'] = self.dataSets[0]['data']
        train['label'] = self.dataSets[0]['label']

        dev['data'] = self.dataSets[1]['data']
        dev['label'] = self.dataSets[1]['label']

        test['data'] = self.dataSets[2]['data']
        test['label'] = self.dataSets[2]['label']

        return train,dev,test

class DataBuilder(data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.label = self.dataset['label']

    def __getitem__(self, index):
        return T.ToTensor()(self.dataset['data'][index]), self.dataset['label'][index]

    def __len__(self):
        return len(self.dataset['data'])
