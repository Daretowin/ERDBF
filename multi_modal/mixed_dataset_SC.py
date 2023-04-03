#By daretowin
import torch.utils.data as data
import numpy as np
from tqdm import tqdm
import warnings


warnings.filterwarnings("ignore")

class MyDatasets(object):
    def __init__(self):
        super().__init__()
        self.dataSets = {}
        self.dataSets[0] = {'data': [], 'label': []}#这里的data是两个单模态加一个多模态也就是混合模态
        self.dataSets[1] = {'data': [], 'label': []}#dev
        self.dataSets[2] = {'data': [], 'label': []}#test

    def splits(self):
        #train data process
        train_data_image = np.load('/server18/wb/agevoxceleb/multimodal_embedding/best_image_emb/SC/train_final_emb.npy',allow_pickle=True)
        train_data_image_0 = np.zeros((137839,1,128))
        train_data_image_1 = np.concatenate((train_data_image,train_data_image_0),axis=2)# image + zero audio

        train_data_audio = np.load('/server18/wb/agevoxceleb/multimodal_embedding/audio_emb/SC/train_em.npy',allow_pickle=True)
        train_data_audio_0 = np.zeros((137839,1,128))
        train_data_audio_1 = np.concatenate((train_data_audio_0,train_data_audio),axis=2)# zero image + audio
        
        train_data_mixed = np.concatenate((train_data_image,train_data_audio),axis=2)
        self.dataSets[0]['data'] = np.concatenate((train_data_image_1,train_data_audio_1,train_data_mixed),axis=0)
        train_single_label = []
        with open('/server18/wb/Age_Work_Summary/Multimodal_Age_Estimation_2/github/multi_modal/data_path/train_label.txt','r') as TR_label:
            train_label_lines = TR_label.readlines()
        for line in train_label_lines:
            line = line[:-1]
            train_single_label.append(int(line))
        train_single_label1 = train_single_label
        train_single_label2 = train_single_label
        self.dataSets[0]['label'] = np.concatenate((train_single_label,train_single_label1,train_single_label2))

        #dev data process
        dev_data_image = np.load('/server18/wb/agevoxceleb/multimodal_embedding/best_image_emb/SC/dev_final_emb.npy',allow_pickle=True)
        dev_data_image_0 = np.zeros((14222,1,128))
        dev_data_image_1 = np.concatenate((dev_data_image,dev_data_image_0),axis=2)

        dev_data_audio = np.load('/server18/wb/agevoxceleb/multimodal_embedding/audio_emb/SC/dev_em.npy',allow_pickle=True)
        dev_data_audio_0 = np.zeros((14222,1,128))
        dev_data_audio_1 = np.concatenate((dev_data_audio_0,dev_data_audio),axis=2)

        dev_data_mixed = np.concatenate((dev_data_image,dev_data_audio),axis=2)
        self.dataSets[1]['data'] = np.concatenate((dev_data_image_1,dev_data_audio_1,dev_data_mixed),axis=0)
        # self.dataSets[1]['data'] = dev_data_mixed
        dev_single_label = []
        with open('/server18/wb/Age_Work_Summary/Multimodal_Age_Estimation_2/github/multi_modal/data_path/dev_label_path.txt','r') as DV_label:
            dev_label_lines = DV_label.readlines()
        for line in dev_label_lines:
            line = line[:-1]
            dev_single_label.append(int(line))
        dev_single_label1 = dev_single_label
        dev_single_label2 = dev_single_label
        self.dataSets[1]['label'] = np.concatenate((dev_single_label,dev_single_label1,dev_single_label2))    
        # self.dataSets[1]['label'] = dev_single_label
        #test data process
        test_data_image = np.load('/server18/wb/agevoxceleb/multimodal_embedding/best_image_emb/SC/test_final_emb.npy',allow_pickle=True)
        test_data_image_0 = np.zeros((16027,1,128))
        test_data_image_1 = np.concatenate((test_data_image,test_data_image_0),axis=2)# image + zero audio

        test_data_audio = np.load('/server18/wb/agevoxceleb/multimodal_embedding/audio_emb/SC/test_em.npy',allow_pickle=True)
        test_data_audio_0 = np.zeros((16027,1,128))
        test_data_audio_1 = np.concatenate((test_data_audio_0,test_data_audio),axis=2)# zero image + audio

        test_data_mixed = np.concatenate((test_data_image,test_data_audio),axis=2)
        self.dataSets[2]['data'] = np.concatenate((test_data_image_1,test_data_audio_1,test_data_mixed),axis=0)
        # self.dataSets[2]['data'] = test_data_mixed
        test_single_label = []
        with open('/server18/wb/Age_Work_Summary/Multimodal_Age_Estimation_2/github/multi_modal/data_path/test_label_path.txt','r') as TS_label:
            test_label_lines = TS_label.readlines()
        for line in test_label_lines:
            line = line[:-1]
            test_single_label.append(int(line))
        test_single_label1 = test_single_label
        test_single_label2 = test_single_label
        self.dataSets[2]['label'] = np.concatenate((test_single_label,test_single_label1,test_single_label2))
        # self.dataSets[2]['label'] = test_single_label

        print(self.dataSets[0]['data'].shape)
        print(len(self.dataSets[0]['label']))
        print(self.dataSets[1]['data'].shape)
        print(len(self.dataSets[1]['label']))
        print(self.dataSets[2]['data'].shape)
        print(len(self.dataSets[2]['label']))
        train = {'data': [],'label': []}
        dev = {'data': [],'label': []}
        test = {'data': [],'label': []}

        train['data'] = self.dataSets[0]['data']
        train['label'] = np.array(self.dataSets[0]['label'])

        dev['data'] = self.dataSets[1]['data']
        dev['label'] = np.array(self.dataSets[1]['label'])

        test['data'] = self.dataSets[2]['data']
        test['label'] = np.array(self.dataSets[2]['label'])
        
        return train,dev,test

class DataBuilder(data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.label = self.dataset['label']

    def __getitem__(self, index):
        return self.dataset['data'][index],self.dataset['label'][index]

    def __len__(self):
        return len(self.dataset['label'])

if __name__ == "__main__":
    train,dev,test = MyDatasets().splits()
    train_dataset = DataBuilder(train)
    dev_dataset = DataBuilder(dev)
    test_dataset = DataBuilder(test)

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=60,
        shuffle=True,)
    
    for batch_idx,(data,label) in enumerate(train_loader):
        pass