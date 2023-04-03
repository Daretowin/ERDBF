#By daretowin
import torch.utils.data as data
import numpy as np
from torchvision.transforms import transforms
import warnings
from PIL import Image
warnings.filterwarnings("ignore")

class MyDatasets(object):
    def __init__(self):
        super().__init__()
        self.dataSets = {}
        self.dataSets[0] = {'data': [], 'label': []}#train
        self.dataSets[1] = {'data': [], 'label': []}#dev
        self.dataSets[2] = {'data': [], 'label': []}#test
        
    def splits(self):
        txt_train = '/server18/wb/Age_Work_Summary/Multimodal_Age_Estimation_2/Multimodal_Age_Estimation/E-Fusion/image_modal/image_train.txt'
        txt_dev = '/server18/wb/Age_Work_Summary/Multimodal_Age_Estimation_2/Multimodal_Age_Estimation/E-Fusion/image_modal/image_dev.txt'
        txt_test = '/server18/wb/Age_Work_Summary/Multimodal_Age_Estimation_2/Multimodal_Age_Estimation/E-Fusion/image_modal/image_test.txt'

        train_label = []
        dev_label = []
        test_label = []

        # train_data = np.load('/server18/wb/agevoxceleb/unimodal_data/image/train_data.npy')
        # train_label = np.load('/server18/wb/agevoxceleb/unimodal_data/image/train_label.npy')

        # dev_data = np.load('/server18/wb/agevoxceleb/unimodal_data/image/dev_data.npy')
        # dev_label = np.load('/server18/wb/agevoxceleb/unimodal_data/image/dev_label.npy')

        test_data = np.load('/server18/wb/agevoxceleb/unimodal_data/image/test_data.npy')
        test_label = np.load('/server18/wb/agevoxceleb/unimodal_data/image/test_label.npy')
        
        # self.dataSets[0]['data'] = train_data
        # self.dataSets[0]['label'] = train_label
        # self.dataSets[1]['data'] = dev_data
        # self.dataSets[1]['label'] = dev_label
        self.dataSets[2]['data'] = test_data
        self.dataSets[2]['label'] = test_label

        train = {'data': [], 'label': []}
        dev = {'data': [], 'label': []}
        test = {'data': [], 'label': []}

        train['data'] = self.dataSets[0]['data']
        train['label'] = np.array(self.dataSets[0]['label'])

        dev['data'] = self.dataSets[1]['data']
        dev['label'] = np.array(self.dataSets[1]['label'])

        test['data'] = self.dataSets[2]['data']
        test['label'] =np.array(self.dataSets[2]['label'])

        print('work done')
        return train,dev,test

class TrainBuilder(data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        train_transform_0 = []

        train_transform_0.append(transforms.RandomHorizontalFlip())
        train_transform_0.append(transforms.RandomGrayscale())
        train_transform_0.append(transforms.ToTensor())
        train_transform_0.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        self.train_transform0 = transforms.Compose(train_transform_0)


    def __getitem__(self, index):
        item0 = self.train_transform0(Image.fromarray(self.dataset['data'][index]))
        return item0, self.dataset['label'][index]

    def __len__(self):

        return len(self.dataset['data'])

class TestBuilder(data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        test_transform_0 = []
        test_transform_0.append(transforms.ToTensor())
        test_transform_0.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        self.test_transform0 = transforms.Compose(test_transform_0)

    def __getitem__(self, index):
        item0 = self.test_transform0(Image.fromarray(self.dataset['data'][index]))
        return item0, self.dataset['label'][index]

    def __len__(self):
        return len(self.dataset['data'])
