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
import time


#files loader class
class Loader(Dataset):
    def __init__(self, split, trasform = None):
        base_dir='./dataset/'

        path =os.path.join(base_dir,'64x64_Faces/{}'.format(split))

        files = os.listdir(path)

        self.filenames = [os.path.join(path,f) for f in files if f.endswith('.png')]

        self.targets = [int(f[0]) for f in files]

        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx])

        if self.transform:
            image = self.transform(image)
        
        return image, self.targets[idx]

from torchvision import transforms


# image to tensor pipeline
transform = transforms.Compose([
    transforms.Lambda(lambda image: image.convert('RGB')),
    transforms.RandomHorizontalFlip(),
    transforms.Resize((64,64),interpolation=Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

#neural network class
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

net = Net()
device = ('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device)
testSet = Loader('train',transform)
start = time.time()

#disable gradients make prediction faster
with torch.no_grad():

    #i'll use this to count wrong predictions
    mistakes = 0

    #i whant to predict 100 images
    for i in range(100):

        #load the pretrained model
        net.load_state_dict(torch.load('model.pth'))

        #load labels and image from my dataset
        image, label = testSet[i]   

        #add extra dimention   
        image = image.unsqueeze(0)

        #convert the data type of the image to float32
        image = torch.clone((image)).to(device)
        
        #make my prediction
        out = net(image)

        #get the maximum number of a probability list of resulst and print it
        porcent = torch.nn.functional.softmax(out, dim=1)[0]*100
        _, index = torch.max(out, dim=1)
        print('input: ',label,'    output: ',index.item(),'   acuracy: ',porcent[index[0]].item())

        #count all wrong predictions
        if label != index.item():
            mistakes +=1

    stop = time.time()
    Etime = stop-start
    
    print('wrong predictions: ',mistakes,'        execution time: ',Etime)