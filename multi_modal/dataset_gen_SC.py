#By daretowin
import torch
import torch.utils.data as data
import numpy as np
import os
from torchvision.transforms import transforms
from tqdm import tqdm
import random
import warnings
import librosa
from PIL import Image
from scipy.io import wavfile
import torchvision as v
from models  import Net1,Net1_AC,Net2,Net2_AC
import importlib
import sys
warnings.filterwarnings("ignore")

class MyDatasets(object):
    def __init__(self):
        super().__init__()
        self.dataSets = {}
        self.dataSets[0] = {'data0': [], 'data1':[], 'label': []}
        self.dataSets[1] = {'data0': [], 'data1':[], 'label': []}

    def loadWAV(self,filename):#
        # Maximum audio length
        max_audio = int(0*160 + 240)
        # Read wav file and convert to torch tensor
        sample_rate, audio  = wavfile.read(filename)
        audiosize = audio.shape[0]
        if audiosize <= max_audio:
            shortage    = max_audio - audiosize + 1 
            audio       = np.pad(audio, (0, shortage), 'wrap')
            audiosize   = audio.shape[0]
        feats = []
        feats.append(audio[:16000*10])
        feat = np.stack(feats,axis=0).astype(np.float)
        feat = np.squeeze(feat,0)
        return feat

    def splits(self):
        txt_train = '/home/great72/great72/work/Multimodal_Age_Estimation/E-Fusion/split_data/train.txt'
        txt_dev = '/home/great72/great72/work/Multimodal_Age_Estimation/E-Fusion/split_data/dev.txt'
        txt_test = '/home/great72/great72/work/Multimodal_Age_Estimation/E-Fusion/split_data/test.txt'
        
        data0_path = '/mnt3/wb/vox2_video/align_images_new/'
        data1_path = '/mnt2/dataset/voxceleb/vox2/dev/aac/'
        
        train_image_path = []
        train_audio_path = []
        dev_image_path = []
        dev_audio_path = []
        test_image_path = []
        test_audio_path = []

        train_image_data = []
        train_audio_data = []
        dev_image_data = []
        dev_audio_data = []
        test_image_data = []
        test_audio_data = []

        train_label = []
        train_dis_label = []
        dev_label = []
        dev_dis_label = []
        test_label = []
        test_dis_label = []

        # # ----处理路径并生成路径
        # with open(txt_train, 'r') as f_train:
        #     lines_train = f_train.readlines()
        
        # for line in tqdm(lines_train):
        #     line = line[:-1]
        #     label = line.split(' ')[1]
        #     path = line.split(' ')[0]
        #     charge = os.path.exists(data0_path+path.split('/')[0]+'/'+path.split('/')[1]+'/'+path.split('/')[2])#判断图像路径是不是存在
        #     charge1 = os.path.exists(data1_path+path.split('/')[0]+'/'+path.split('/')[1]+'/'+path.split('/')[2]+'.wav')#判断音频路径是不是存在
        #     if charge==True and charge1==True:
        #         image_item = os.listdir(data0_path+path.split('/')[0]+'/'+path.split('/')[1]+'/'+path.split('/')[2])#图像路径下的图片
        #         lenth = len(image_item)
        #         if lenth > 2:
                    # # 添加3张图像路径
                    # train_image_path.append(data0_path+path.split('/')[0]+'/'+path.split('/')[1]+'/'+path.split('/')[2]+'/'+image_item[0])
                    # train_image_path.append(data0_path+path.split('/')[0]+'/'+path.split('/')[1]+'/'+path.split('/')[2]+'/'+image_item[1])
                    # train_image_path.append(data0_path+path.split('/')[0]+'/'+path.split('/')[1]+'/'+path.split('/')[2]+'/'+image_item[2])
                    # #添加音频路径
                    # train_audio_path.append(data1_path+path.split('/')[0]+'/'+path.split('/')[1]+'/'+path.split('/')[2]+'.wav')
                    #添加标签
                    # train_label.append(int(label))
        # for i1 in train_image_path:
        #     with open('/home/great72/great72/work/Multimodal_Age_Estimation/E-Fusion/datapath/train_image_path.txt','a') as x1:
        #         x1.write(i1+'\n')
        # for i2 in train_audio_path:
        #     with open('/home/great72/great72/work/Multimodal_Age_Estimation/E-Fusion/datapath/train_audio_path.txt','a') as x2:
        #         x2.write(i2+'\n')
        # for i3 in train_label:
        #     with open('/home/great72/great72/work/Multimodal_Age_Estimation/E-Fusion/datapath/train_label.txt','a') as x3:
        #         x3.write(str(i3)+'\n')
        # print('process done')

        # #----读取路径，保存数据 image
        # with open('/home/great72/great72/work/Multimodal_Age_Estimation/E-Fusion/datapath/train_image_path.txt','r') as f_image:
        #     image_lines = f_image.readlines()
        # for line in tqdm(image_lines[275678:]):
        #     line = line[:-1]
        #     image = Image.open(line)
        #     image = np.array(image)
        #     train_image_data.append(image)
        # np.save('/mnt3/wb/agevoxceleb/multimodal_embedding/image_data/train_image_data_part2.npy',train_image_data)

        # # ----读取路径，保存数据 audio
        # with open('/home/great72/great72/work/Multimodal_Age_Estimation/E-Fusion/datapath/train_audio_path.txt','r') as f_audio:
        #     audio_lines = f_audio.readlines()
        # for line in tqdm(audio_lines[68919:]):
        #     line = line[:-1]
        #     audio = self.loadWAV(line)
        #     train_audio_data.append(audio)
        # np.save('/mnt3/wb/agevoxceleb/multimodal_embedding/audio_data/train_audio_data_part1.npy',train_audio_data)

        # #----处理路径并保存
        # with open(txt_test, 'r') as f_test:
        #     lines_test = f_test.readlines()
        
        # for line in tqdm(lines_test):
        #     line = line[:-1]
        #     label = line.split(' ')[1]
        #     path = line.split(' ')[0]
        #     charge = os.path.exists(data0_path+path.split('/')[0]+'/'+path.split('/')[1]+'/'+path.split('/')[2])#判断图像路径是不是存在
        #     charge1 = os.path.exists(data1_path+path.split('/')[0]+'/'+path.split('/')[1]+'/'+path.split('/')[2]+'.wav')#判断音频路径是不是存在
        #     if charge==True and charge1==True:
        #         image_item = os.listdir(data0_path+path.split('/')[0]+'/'+path.split('/')[1]+'/'+path.split('/')[2])#图像路径下的图片
        #         lenth = len(image_item)
        #         if lenth > 2:
        #             #添加3张图像路径
        #             test_image_path.append(data0_path+path.split('/')[0]+'/'+path.split('/')[1]+'/'+path.split('/')[2]+'/'+image_item[0])
        #             test_image_path.append(data0_path+path.split('/')[0]+'/'+path.split('/')[1]+'/'+path.split('/')[2]+'/'+image_item[1])
        #             test_image_path.append(data0_path+path.split('/')[0]+'/'+path.split('/')[1]+'/'+path.split('/')[2]+'/'+image_item[2])
        #             #添加音频路径
        #             test_audio_path.append(data1_path+path.split('/')[0]+'/'+path.split('/')[1]+'/'+path.split('/')[2]+'.wav')
        #             #添加标签
        #             test_label.append(int(label))
        # for i1 in test_image_path:
        #     with open('/home/great72/great72/work/Multimodal_Age_Estimation/E-Fusion/datapath/test_image_path.txt','a') as x1:
        #         x1.write(i1+'\n')
        # for i2 in test_audio_path:
        #     with open('/home/great72/great72/work/Multimodal_Age_Estimation/E-Fusion/datapath/test_audio_path.txt','a') as x2:
        #         x2.write(i2+'\n')
        # for i3 in test_label:
        #     with open('/home/great72/great72/work/Multimodal_Age_Estimation/E-Fusion/datapath/test_label_path.txt','a') as x3:
        #         x3.write(str(i3)+'\n')
        # print('process done')

        # # ----读取路径，保存数据 image
        # with open('/home/great72/great72/work/Multimodal_Age_Estimation/E-Fusion/datapath/test_image_path.txt','r') as f_image:
        #     image_lines = f_image.readlines()
        # for line in tqdm(image_lines):
        #     line = line[:-1]
        #     image = Image.open(line)
        #     image = np.array(image)
        #     test_image_data.append(image)
        # np.save('/mnt3/wb/agevoxceleb/multimodal_embedding/image_data/test_image_data.npy',test_image_data)

        # # ----读取路径，保存数据 audio
        # with open('/home/great72/great72/work/Multimodal_Age_Estimation/E-Fusion/datapath/test_audio_path.txt','r') as f_audio:
        #     audio_lines = f_audio.readlines()
        # for line in tqdm(audio_lines):
        #     line = line[:-1]
        #     audio = self.loadWAV(line)
        #     test_audio_data.append(audio)
        # np.save('/mnt3/wb/agevoxceleb/multimodal_embedding/audio_data/test_audio_data.npy',test_audio_data)


        # #----处理路径并保存
        # with open(txt_dev, 'r') as f_dev:
        #     lines_dev = f_dev.readlines()
        
        # for line in tqdm(lines_dev):
        #     line = line[:-1]
        #     label = line.split(' ')[1]
        #     path = line.split(' ')[0]
        #     charge = os.path.exists(data0_path+path.split('/')[0]+'/'+path.split('/')[1]+'/'+path.split('/')[2])#判断图像路径是不是存在
        #     charge1 = os.path.exists(data1_path+path.split('/')[0]+'/'+path.split('/')[1]+'/'+path.split('/')[2]+'.wav')#判断音频路径是不是存在
        #     if charge==True and charge1==True:
        #         image_item = os.listdir(data0_path+path.split('/')[0]+'/'+path.split('/')[1]+'/'+path.split('/')[2])#图像路径下的图片
        #         lenth = len(image_item)
        #         if lenth > 2:
        #             #添加3张图像路径
        #             dev_image_path.append(data0_path+path.split('/')[0]+'/'+path.split('/')[1]+'/'+path.split('/')[2]+'/'+image_item[0])
        #             dev_image_path.append(data0_path+path.split('/')[0]+'/'+path.split('/')[1]+'/'+path.split('/')[2]+'/'+image_item[1])
        #             dev_image_path.append(data0_path+path.split('/')[0]+'/'+path.split('/')[1]+'/'+path.split('/')[2]+'/'+image_item[2])
        #             #添加音频路径
        #             dev_audio_path.append(data1_path+path.split('/')[0]+'/'+path.split('/')[1]+'/'+path.split('/')[2]+'.wav')
        #             #添加标签
        #             dev_label.append(int(label))
        # for i1 in dev_image_path:
        #     with open('/home/great72/great72/work/Multimodal_Age_Estimation/E-Fusion/datapath/dev_image_path.txt','a') as x1:
        #         x1.write(i1+'\n')
        # for i2 in dev_audio_path:
        #     with open('/home/great72/great72/work/Multimodal_Age_Estimation/E-Fusion/datapath/dev_audio_path.txt','a') as x2:
        #         x2.write(i2+'\n')
        # for i3 in dev_label:
        #     with open('/home/great72/great72/work/Multimodal_Age_Estimation/E-Fusion/datapath/dev_label_path.txt','a') as x3:
        #         x3.write(str(i3)+'\n')
        # print('process done')

        # # ----读取路径，保存数据 image
        # with open('/home/great72/great72/work/Multimodal_Age_Estimation/E-Fusion/datapath/dev_image_path.txt','r') as f_image:
        #     image_lines = f_image.readlines()
        # for line in tqdm(image_lines):
        #     line = line[:-1]
        #     image = Image.open(line)
        #     image = np.array(image)
        #     dev_image_data.append(image)
        # np.save('/mnt3/wb/agevoxceleb/multimodal_embedding/image_data/dev_image_data.npy',dev_image_data)

        # # ----读取路径，保存数据 audio
        # with open('/home/great72/great72/work/Multimodal_Age_Estimation/E-Fusion/datapath/dev_audio_path.txt','r') as f_audio:
        #     audio_lines = f_audio.readlines()
        # for line in tqdm(audio_lines):
        #     line = line[:-1]
        #     audio = self.loadWAV(line)
        #     dev_audio_data.append(audio)
        # np.save('/mnt3/wb/agevoxceleb/multimodal_embedding/audio_data/dev_audio_data.npy',dev_audio_data)




        # # # # #----加载训练数据
        # train_image_data_part0 = np.load('/mnt3/wb/agevoxceleb/multimodal_embedding/image_data/train_image_data_part0.npy',allow_pickle=True)
        # train_image_data_part1 = np.load('/mnt3/wb/agevoxceleb/multimodal_embedding/image_data/train_image_data_part1.npy',allow_pickle=True)
        # train_image_data_part2 = np.load('/mnt3/wb/agevoxceleb/multimodal_embedding/image_data/train_image_data_part2.npy',allow_pickle=True)
        
        # # train_audio_data_part0 = np.load('/mnt3/wb/agevoxceleb/multimodal_embedding/audio_data/train_audio_data_part0.npy',allow_pickle=True)
        # # train_audio_data_part1 = np.load('/mnt3/wb/agevoxceleb/multimodal_embedding/audio_data/train_audio_data_part1.npy',allow_pickle=True)

        # train_image_data = np.concatenate((train_image_data_part0,train_image_data_part1,train_image_data_part2),axis=0)
        # # train_audio_data = np.concatenate((train_audio_data_part0,train_audio_data_part1),axis=0)

        # with open('/home/great72/great72/work/Multimodal_Age_Estimation/E-Fusion/datapath/train_label.txt','r') as TR_label:
        #     train_label_lines = TR_label.readlines()
        # for line in train_label_lines:
        #     line = line[:-1]
        #     line1 = line
        #     line2 = line
        #     line3 = line 
        #     train_label.append(int(line1))
        #     train_label.append(int(line2))
        #     train_label.append(int(line3))
            
        # # # # ----加载测试数据
        # test_image_data = np.load('/mnt3/wb/agevoxceleb/multimodal_embedding/image_data/test_image_data.npy',allow_pickle=True)
        # # test_audio_data = np.load('/mnt3/wb/agevoxceleb/multimodal_embedding/audio_data/test_audio_data.npy',allow_pickle=True)
        # with open('/home/great72/great72/work/Multimodal_Age_Estimation/E-Fusion/datapath/test_label_path.txt','r') as TS_label:
        #     test_label_lines = TS_label.readlines()
        # for line in test_label_lines:
        #     line = line[:-1]
        #     line1 = line
        #     line2 = line
        #     line3 = line
        #     test_label.append(int(line1))
        #     test_label.append(int(line2))
        #     test_label.append(int(line3))

        # # # ----加载开发数据
        # dev_image_data = np.load('/mnt3/wb/agevoxceleb/multimodal_embedding/image_data/dev_image_data.npy',allow_pickle=True)
        # # dev_audio_data = np.load('/mnt3/wb/agevoxceleb/multimodal_embedding/audio_data/dev_audio_data.npy',allow_pickle=True)
        # with open('/home/great72/great72/work/Multimodal_Age_Estimation/E-Fusion/datapath/dev_label_path.txt','r') as DV_label:
        #     dev_label_lines = DV_label.readlines()
        # for line in dev_label_lines:
        #     line = line[:-1]
        #     line1 = line
        #     line2 = line
        #     line3 = line
        #     dev_label.append(int(line1))
        #     dev_label.append(int(line2))
        #     dev_label.append(int(line3))


        train = {'data0': [], 'data1':[], 'label': []}
        dev = {'data0': [], 'data1':[], 'label': []}
        test = {'data0': [], 'data1':[], 'label': []}

        train['data0'] = train_image_data
        # train['data1'] = train_audio_data
        train['label'] = np.array(train_label)

        dev['data0'] = dev_image_data
        # dev['data1'] = dev_audio_data
        dev['label'] = np.array(dev_label)

        test['data0'] = test_image_data
        # test['data1'] = test_audio_data
        test['label'] = np.array(test_label)

        print('work done')
        return train,dev,test
class TrainBuilder(data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
        train_transform = []
        train_transform.append(transforms.RandomHorizontalFlip())
        train_transform.append(transforms.RandomGrayscale())
        train_transform.append(transforms.ToTensor())
        train_transform.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        self.train_transform2 = transforms.Compose(train_transform)

    def __getitem__(self, index):
        item = self.train_transform2(Image.fromarray(self.dataset['data0'][index]))

        return item,self.dataset['label'][index]

    def __len__(self):
        return len(self.dataset['data0'])

class TestBuilder(data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
        test_transform = []
        test_transform.append(transforms.ToTensor())
        test_transform.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        self.test_transform2 = transforms.Compose(test_transform)
        
    def __getitem__(self, index):
        item = self.test_transform2(Image.fromarray(self.dataset['data0'][index]))
        return item, self.dataset['label'][index]

    def __len__(self):
        return len(self.dataset['data0'])

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] ='1'

    train,dev,test = MyDatasets().splits()
    train_dataset = TrainBuilder(train)
    dev_dataset = TestBuilder(dev)
    test_dataset = TestBuilder(test)
    
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False)    
    dev_loader = data.DataLoader(
        dev_dataset,
        batch_size=1,
        shuffle=False)   
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False)
    
    # # ---- image inference 用来生成384 or 192维图像向量
    # resnet = v.models.resnet50(pretrained=False)
    # model = Net1(resnet)
    # model.load_state_dict(torch.load('/home/great72/great72/work/Multimodal_Age_Estimation/E-Fusion/image_modal/saved_models/SC0001_best.pt'))
    # model = model.cuda()
    # model.eval()
    # imageout_train = []
    # imageout_dev = []
    # imageout_test = []

    # for batch_idx,(data0,label) in tqdm(enumerate(train_loader)):
    #     data0 = data0.cuda()
    #     label = label.cuda()
    #     [x1,x2],out = model(data0)
    #     pre = out.cpu().detach().numpy()
    #     imageout_train.append(pre)
    
    # np.save('/mnt3/wb/agevoxceleb/multimodal_embedding/best_image_emb/SC/train_emb.npy',imageout_train)

    # for batch_idx,(data0,label) in tqdm(enumerate(dev_loader)):
    #     data0 = data0.cuda()
    #     label = label.cuda()
    #     [x1,x2],out = model(data0)
    #     pre = out.cpu().detach().numpy()
    #     imageout_dev.append(pre)
    
    # np.save('/mnt3/wb/agevoxceleb/multimodal_embedding/best_image_emb/SC/dev_emb.npy',imageout_dev)

    # for batch_idx,(data0,label) in tqdm(enumerate(test_loader)):
    #     data0 = data0.cuda()
    #     label = label.cuda()
    #     [x1,x2],out = model(data0)
    #     pre1 = out.cpu().detach().numpy()
    #     imageout_test.append(pre1)
    
    # np.save('/mnt3/wb/agevoxceleb/multimodal_embedding/best_image_emb/SC/test_emb.npy',imageout_test)

    # ---- image 向量三张取平均
    train_3 = np.load('/mnt3/wb/agevoxceleb/multimodal_embedding/best_image_emb/SC/train_emb.npy',allow_pickle=True)
    dev_3 = np.load('/mnt3/wb/agevoxceleb/multimodal_embedding/best_image_emb/SC/dev_emb.npy',allow_pickle=True)
    test_3 = np.load('/mnt3/wb/agevoxceleb/multimodal_embedding/best_image_emb/SC/test_emb.npy',allow_pickle=True)
    train_1 = []
    dev_1 = []
    test_1 = []

    for i in range(len(train_3)):
        if i%3==0:
            train_1.append(np.mean(train_3[i:i+3],axis=0))

    for i in range(len(dev_3)):
        if i%3==0:
            dev_1.append(np.mean(dev_3[i:i+3],axis=0))

    for i in range(len(test_3)):
        if i%3==0:
            test_1.append(np.mean(test_3[i:i+3],axis=0))

    np.save('/mnt3/wb/agevoxceleb/multimodal_embedding/best_image_emb/SC/train_final_emb.npy',train_1)
    np.save('/mnt3/wb/agevoxceleb/multimodal_embedding/best_image_emb/SC/dev_final_emb.npy',dev_1)
    np.save('/mnt3/wb/agevoxceleb/multimodal_embedding/best_image_emb/SC/test_final_emb.npy',test_1)


    # # ---- audio inference 用来生成192维音频向量 然后通过audio_dnn 生成384维 or 192维音频向量
    # sys.path.append('/workspace/LOGS_OUTPUT/server9_nvme1/ASV_LOGS_202102/train_dist')
    # sys.path.append('/workspace/GREAT_ASV_system/Model_exp')

    # model_path = '/nvme1/zhiyong/ASV_LOGS_202102/train_logs_201120/xvector(vox2)/model/model000000134.model'

    # SpeakerNetModel = importlib.import_module('audio_models.'+'X_vector').__getattribute__('MainModel')
    # S = SpeakerNetModel(n_mels=40, nOut=192, spec_aug=False)
    # loaded_state = torch.load(model_path, map_location="cuda:0")

    # self_state = S.state_dict()

    # for name, param in loaded_state['model'].items():
    #     origname = name

    #     ## pass spk clf weight
    #     if '__L__' in name:
    #         print('pass __L__ classerfier W')
    #         continue

    #     ## pass DA weight
    #     if 'DA_module' in name:
    #         print('pass DA_module params:'+name)
    #         continue

    #     if name not in self_state:
    #         name = name.replace("__S__.", "")

    #         if name not in self_state:
    #             print("#%s is not in the model."%origname)
    #             continue

    #     if self_state[name].size() != loaded_state['model'][origname].size():
    #         print("#Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state['model'][origname].size()))
    #         continue

    #     self_state[name].copy_(param)
    # S.cuda()
    # S.eval()
    # trem = []
    # dvem = []
    # tsem = []

    # model = Net2()
    # model.load_state_dict(torch.load('/home/great72/great72/work/Multimodal_Age_Estimation/E-Fusion/audio_modal/saved_models/SC_best.pt'))
    # model.cuda()
    # model.eval()

    # for batch_idx,(data1,label) in tqdm(enumerate(train_loader)):
    #     data1 = data1.numpy()
    #     raw_inp = torch.FloatTensor(data1).cuda()
    #     # ref_feat = S.forward(raw_inp).detach().cpu().numpy()
    #     ref_feat = S.forward(raw_inp)
    #     [x1,x2],out = model(ref_feat)
    #     out = out.detach().cpu().numpy()
    #     trem.append(out)

    # np.save('/mnt3/wb/agevoxceleb/multimodal_embedding/audio_emb/SC/train_em.npy',trem)

    # for batch_idx,(data1,label) in tqdm(enumerate(dev_loader)):
    #     data1 = data1.numpy()
    #     raw_inp = torch.FloatTensor(data1).cuda()
    #     # ref_feat = S.forward(raw_inp).detach().cpu().numpy()
    #     ref_feat = S.forward(raw_inp)
    #     [x1,x2],out = model(ref_feat)
    #     out = out.detach().cpu().numpy()
    #     dvem.append(out)

    # np.save('/mnt3/wb/agevoxceleb/multimodal_embedding/audio_emb/SC/dev_em.npy',dvem)
    

    # for batch_idx,(data1,label) in tqdm(enumerate(test_loader)):
    #     data1 = data1.numpy()
    #     raw_inp = torch.FloatTensor(data1).cuda()
    #     # ref_feat = S.forward(raw_inp).detach().cpu().numpy()
    #     ref_feat = S.forward(raw_inp)
    #     [x1,x2],out = model(ref_feat)
    #     out = out.detach().cpu().numpy()
    #     tsem.append(out)

    # np.save('/mnt3/wb/agevoxceleb/multimodal_embedding/audio_emb/SC/test_em.npy',tsem)
    

    
