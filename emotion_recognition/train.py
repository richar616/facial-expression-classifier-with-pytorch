import os
import numpy as np 
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset, DataLoader 
from matplotlib import pyplot as plt
import torchvision.transforms as transforms

import torch.optim as optim

#this class will feed my neural network with images
class Loader(Dataset):
    def __init__(self, split, trasform = None):
        base_dir='./dataset/'

        path =os.path.join(base_dir,'64x64_Faces/{}'.format(split))

        files = os.listdir(path)

        self.filenames = [os.path.join(path,f) for f in files if f.endswith('.png')]

        self.targets = [int(f[0]) for f in files]# I ussig the first number of each image as a label

        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx])

        if self.transform:
            image = self.transform(image)
        
        return image, self.targets[idx]

from torchvision import transforms

#transform pipeline
transform = transforms.Compose([
    transforms.Lambda(lambda image: image.convert('RGB')),
    transforms.RandomHorizontalFlip(),
    transforms.Resize((64,64),interpolation=Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

#instace my dataset and dataloader
trainSet = Loader('train',transform)
trainLoader = DataLoader(trainSet, batch_size=1)
dataLoader = {'train':trainLoader}

#my neural network class
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=9,kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=9,out_channels=12,kernel_size=5)
        self.fc1 = nn.Linear(2028,1000)
        self.fc2 = nn.Linear(1000,500)
        self.fc3 = nn.Linear(500,8)
        self.pool = nn.MaxPool2d(2)

    def forward(self,x):
        x = self.pool(F.relu( self.conv1(x)))
        x = self.pool(F.relu( self.conv2(x)))

        x = x.view(-1,2028)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        x = F.log_softmax(x,dim=1)
        return x

#use gpu if available if not cpu
device = ('cuda' if torch.cuda.is_available() else 'cpu')


net = Net()
net.to(device)


from torch import optim 

loss_fn = nn.NLLLoss()

optimizer = optim.Adam(net.parameters(),lr=0.001)


epocs = 10

import time
start = time.time()

#training loop
for epoch in range(epocs):
    
    for inputs, targets in dataLoader['train']:

        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()

        outputs = net(inputs)
        
        loss = loss_fn(outputs, targets) 

        loss.backward()
 
        optimizer.step()

        print('loss: ',loss.item(),'      epoc: ',epoch)

stop = time.time()
Etime = stop-start
print('execution time: ',Etime)

#save the trained model
torch.save(net.state_dict(),'model.pth')
