import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
import time
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from torchmetrics.functional import r2_score

import warnings
warnings.filterwarnings('ignore')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path = r'D:\Data\692 Working Space'


class MyDataset(Dataset):
    def __init__(self,img_path, transform = None):
        super(MyDataset, self).__init__()
        self.root = img_path
        self.Porosity =self.root + '\\' + 'Porosity_sq1000200'
        self.Thickness =self.root + '\\' + 'Thickness_sq1000200'
        self.Pressure = self.root + '\\' + 'Pressure_sq1000200'


        # Structured Features
        Structured_data = pd.read_csv(self.root+ '\\'+'data_total_20_features.csv')
        WellID = list(Structured_data['UWI'])
        # normalization
        Structured_data_nor = Structured_data.drop(['UWI'],axis=1 )
        columns =  Structured_data_nor.columns
        minMax = MinMaxScaler()
        X = minMax.fit_transform(Structured_data_nor)
        Structured_data_n = pd.DataFrame(data =X, columns= columns)
        Target_production_n = np.array(Structured_data_n['NGE_12'])
        Structured_data_n['UWI'] = Structured_data['UWI']
        Target_production = []
        for i in WellID:
                Target_production_ = copy.deepcopy(Structured_data_n.loc[Structured_data_n['UWI']== i])
                Target_production_ = Target_production_['NGE_12']
                Target_production.append(np.array(Target_production_,dtype=float))
        self.Target = Target_production
        Structured_data_n.drop(['NGE_12'],axis=1,inplace= True)
        Structured_features = []
        for i in WellID:
                # features
                Structured_feature = copy.deepcopy(Structured_data_n.loc[Structured_data_n['UWI']== i])
                Structured_feature.drop(columns = 'UWI',inplace = True)
                Structured_features.append(np.array(Structured_feature,dtype=float))
        self.Well_Structured_features =  Structured_features

        self.WellID = WellID
        self.transform = transform

        f2 = open(self.root+ '\\'+'ID_Porosity.txt','r')
        data_Porosity = f2.readlines()
        imgs_Porosity = []
        for line in data_Porosity:
            line_Porosity = line.rstrip()
            word_Porosity = line.split()
            # print('Porperty:',word_Porosity[1],'WellID:',
            #       str(word_Porosity[0]).replace('_','/'),
            # print( line_Porosity)
            imgs_Porosity.append(os.path.join(self.Porosity,line_Porosity[:-5]+'.jpg'))
        self.Porosity_img = imgs_Porosity

        f3 = open(self.root+ '\\'+'ID_Thickness.txt','r')
        data_Thickness = f3.readlines()
        imgs_Thickness = []
        for line in data_Thickness:
            line_Thickness = line.rstrip()
            word_Thickness = line.split()
            imgs_Thickness.append(os.path.join(self.Thickness,line_Thickness[:-5]+'.jpg'))
        self.Thickness_img = imgs_Thickness

        f4 = open(self.root+ '\\'+'ID_Pressure.txt','r')
        data_Pressure = f4.readlines()
        imgs_Pressure = []
        for line in data_Pressure:
            line_Pressure = line.rstrip()
            word_Pressure = line.split()
            imgs_Pressure.append(os.path.join(self.Pressure,line_Pressure[:-5]+'.jpg'))
        self.Pressure_img = imgs_Pressure

        # Preloading images to GPU
        self.to_tensor = transforms.ToTensor()

        self.Porosity_images = [Image.open(img_path).convert('L') for img_path in self.Porosity_img]
        self.Thickness_images = [Image.open(img_path).convert('L') for img_path in self.Thickness_img]
        self.Pressure_images = [Image.open(img_path).convert('L') for img_path in self.Pressure_img]
    def __len__(self):

        return len(self.WellID)

    def __getitem__(self, item):
        Structured_features = self.Well_Structured_features[item]
        WellID = self.WellID[item]
        Target = self.Target[item]

        img_Porosity = self.Porosity_images[item]
        img_Thickness = self.Thickness_images[item]
        img_Pressure = self.Pressure_images[item]

        if self.transform is not None:
            img_Porosity = self.transform(img_Porosity).to(device)
            img_Thickness = self.transform(img_Thickness).to(device)
            img_Pressure = self.transform(img_Pressure).to(device)

        Distribution =    torch.cat((
            img_Porosity,
            img_Thickness,
            img_Pressure), dim=0)
        return WellID ,Target, Structured_features, Distribution

class ML(nn.Module): #   MMML CNN
    def __init__(self,patch_size,patch_embedding_dim,ic,image_h,image_w):
        super(ML, self).__init__()
        self.patch_size = patch_size   #   patch_size
        self.patch_embedding_dim = patch_embedding_dim #   the assumed features of the patch
        self.ic = ic   #   the channel of the image
        self.image_h = image_h  #   the height of the image
        self.image_w = image_w #   the width of the image
        self.pool = 2
        # self.conv2int = ((((image_h-patch_size+1)//self.pool-patch_size+1)//self.pool-patch_size+1))*((((image_h-patch_size+1)//self.pool-patch_size+1)//self.pool-patch_size+1))
        self.conv2int = (
                         (
                           (((image_h - patch_size + 1) //
                            self.pool - patch_size + 1) //   # pooling #1
                            self.pool - patch_size + 1) //   # pooling #2
                            self.pool - patch_size + 1       # pooling #3
                          ) **2
                         )
        # CNN for image
        self.conv1 = nn.Conv2d(ic,patch_embedding_dim,(patch_size,patch_size)) #   in: batch *ic *image_h *image_w out: batch*patch_embedding_dim*(image_h-patch_size+1)*(image_w-patch_size+1)
        self.conv2 = nn.Conv2d(patch_embedding_dim,patch_embedding_dim,(patch_size,patch_size))
        self.conv3 = nn.Conv2d(patch_embedding_dim, patch_embedding_dim, (patch_size, patch_size))
        self.conv4 = nn.Conv2d(patch_embedding_dim,1,(patch_size,patch_size))
        self.fc1 = nn.Linear(self.conv2int,128*4*self.ic )
        self.fc2 = nn.Linear(128*4*self.ic,128*2*self.ic)
        self.fc3 = nn.Linear(128*2*self.ic, 128*self.ic)
        self.fc4 = nn.Linear(128*self.ic,1)

        # NN for tabular
        self.liner1 = nn.Linear(20, 512)
        self.liner2 = nn.Linear(512, 256)
        self.liner3 = nn.Linear(256, 128)
        self.liner_out = nn.Linear(128, 90)

        # merge
        self.liner_ME1 = nn.Linear(100, 64)
        self.liner_ME2 = nn.Linear(64, 32)
        self.liner_ME3 = nn.Linear(32, 16)
        self.liner_ME4 = nn.Linear(16, 10)

    def forward(self,image,tabular):
        # CNN
        image_size = image.shape[0]
        x1 = F.relu(self.conv1(image))
        x1 = F.max_pool2d(x1,(self.pool,self.pool))
        x1 = F.relu(self.conv2(x1))
        x1 = F.max_pool2d(x1,(self.pool,self.pool))
        x1 = F.relu(self.conv3(x1))
        x1 = F.max_pool2d(x1,(self.pool,self.pool))
        x1 = F.relu(self.conv4(x1))
        x1 = x1.view(image_size,-1)
        x1 = F.relu(self.fc1(x1))
        x1 = F.relu(self.fc2(x1))
        x1 = F.relu(self.fc3(x1))
        x1 = F.relu(self.fc4(x1))
        # x1 = self.fc3(x1)
        # print(x1.shape)

        # NN
        tabular_size = tabular.shape[0]
        x2 = tabular.view(tabular_size,-1)
        x2 = F.relu(self.liner1(x2))
        x2 = F.relu(self.liner2(x2))
        x2 = F.relu(self.liner3(x2))
        x2 = F.relu(self.liner_out(x2))
        # x2 = self.liner_out(x2)
        # print(x2.shape)

        # merge
        x = torch.cat((x1,x2),dim=1)
        x = F.relu(self.liner_ME1(x))
        x = F.relu(self.liner_ME2(x))
        x = F.relu(self.liner_ME3(x))
        x = self.liner_ME4(x)
        #print(x)
        return x

generator1 = torch.Generator()
dataset = MyDataset(path)
batch_size_tr = 477
batch_size_te = 204
feature_number = 20
train_set, test_set = torch.utils.data.random_split(dataset, [0.7, 0.3], generator=generator1)
transforms_ = transforms.Compose([transforms.Resize((400,400)),transforms.ToTensor()])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size_tr, shuffle=False)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size_te, shuffle=False)

epoch_ = 200
Total_history = pd.DataFrame()
def train(train_loader,model,loss_fn,optimizer,feature_number):
    model.train()
    train_mse = 0
    train_r2 = 0
    for batch_idx, data in enumerate(train_loader):
        WellID, Target, Structured_features,Distribution = data
        # print(len(WellID), Target.shape, Structured_features.shape)
        Target, Structured_features,Distribution = Target.to(device), Structured_features.to(device),Distribution.to(device)
        Structured_features_new = Structured_features.reshape((batch_size_tr, 1, feature_number)).float()
        Target_new = Target.reshape((batch_size_tr, 1)).float()
        optimizer.zero_grad()
        out = model(Distribution,Structured_features_new)
        loss = loss_fn(out,Target_new)
        train_mse = float(loss)
        train_r2 = r2_score(out, Target_new).to(device)
        train_mse_history.append(train_mse)
        train_r2_history.append(float(train_r2.item()))
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f'Train MSE: {train_mse:>.3f},Train R2: {train_r2:>.3f}')
    return train_mse, train_r2

def test(test_loader,model,loss_fn,feature_number):
    model.eval()
    test_mse = 0
    test_r2 = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            WellID, Target, Structured_features, Distribution  = data
            Target, Structured_features,Distribution = Target.to(device), Structured_features.to(device),Distribution.to(device)
            Structured_features_new = Structured_features.reshape((batch_size_te, 1, feature_number)).float()
            Target_new = Target.reshape((batch_size_te, 1)).float()
            out_ = model(Distribution ,Structured_features_new)
            loss_ = loss_fn(out_, Target_new)
            test_mse = float(loss_)
            test_r2 = r2_score(out_, Target_new).to(device)
            test_mse_history.append(test_mse)
            test_r2_history.append(float(test_r2.item()))

    print(f'Test MSE: {test_mse:>.3f},Test R2: {test_r2:>.3f}')
    return test_mse, test_r2


for loop in range(0,50):
    time1 = time.time()
    model = ML(5,8,3,400,400).to(device)
    criterion = torch.nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    generator1 = torch.Generator()
    dataset = MyDataset(path,transform = transforms_)
    train_set, test_set = torch.utils.data.random_split(dataset, [0.7, 0.3], generator=generator1)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size_tr, shuffle=False)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size_te, shuffle=False)

    train_mse_history = []
    train_r2_history = []
    test_mse_history = []
    test_r2_history = []

    history = pd.DataFrame()
    for epoch in range(epoch_):
        print('Epoch:', epoch)

        train_mse, train_r2 = train(train_loader, model, criterion, optimizer, feature_number)
        test_mse, test_r2 = test(test_loader, model, criterion, feature_number)

        history = pd.DataFrame({'train_mse':train_mse_history,
                        'train_r2':train_r2_history,
                        'test_mse':test_mse_history,
                        'test_r2':test_r2_history
                        })

        sweet_point = history[history['test_r2'] == max(history['test_r2'])]

        Single_history = pd.DataFrame({
                        'epoch': range(0,epoch_),
                        'train_mse':train_mse_history,
                        'train_r2':train_r2_history,
                        'test_mse':test_mse_history,
                        'test_r2':test_r2_history,
                        })
        Single_history.to_csv('20241209_Single_history.csv',index=False)

        Total_history = Total_history.append({
                        'Loop':loop,
                        'train_mse':np.float64(sweet_point['train_mse']),
                        'train_r2':np.float64(sweet_point['train_r2']),
                        'test_mse':np.float64(sweet_point['test_mse']),
                        'test_r2':np.float64(sweet_point['test_r2'])},ignore_index=True)

Total_history.to_csv('20241209_Total_history',index=False)