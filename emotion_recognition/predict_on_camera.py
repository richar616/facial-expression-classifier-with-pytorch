import torch
import torch.nn as nn
import torch.nn.functional as F

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader 
import torchvision.transforms as transforms

import torch.optim as optim

import cv2

#transform tensor to image
T2 = transforms.ToPILImage()

#image to tensor pipeline
transform = transforms.Compose([
    transforms.Lambda(lambda image: image.convert('RGB')),
    transforms.RandomHorizontalFlip(),
    transforms.Resize((64,64),interpolation=Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

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

#load my pretrained model and put it on gpu if available
PATH = './model.pth'
device = ('cuda' if torch.cuda.is_available() else 'cpu')
net = Net().to(device)


try:
    net.load_state_dict(torch.load(PATH))
    net.to(device)
except:
    print('you don\'t have a pretrained model yet')


def PREDICT_CAM():
    with torch.no_grad():

        #load the cascade file
        face_cascade = cv2.CascadeClassifier('./cascade/haarcascade_frontalface_default.xml')
        
        #load the label
        with open('labels(ES).txt') as f:
            labels = [line.strip() for line in f.readlines()]

        net.eval()

        #start the camera
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()

            #if i whant to detect faces the first thing is convert the image to gray
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # i ca play with those numbers to improve
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
            
            #quick example of how to print the boxes values
            def see_boxes():
                for(x,y,w,h) in faces: #get boxes values and store in varibles
                    
                    ROI_face = frame[y:y+h, x:x+w] #extract the face from the image

                    img = T2(ROI_face) #conver the image to pil

                    img = transform(img) #resize, transform to tensor, normalize, etc
                    
                    img = torch.clone((img))#this is better then this img = torch.tensor((img),dtype= torch.float32)

                    img = img.unsqueeze(0) #add extra dimention

                    img = img.to(device) #pas the image to the gpu

                    out = net(img) #predict who is in the image

                    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100 #get the acuracy of the prediction

                    _, index = torch.max(out,dim=1) #get the best value in my predictions list
  
                    label, acuracy = labels[index.item()], str(percentage[index[0]].item()) #pass the accuracy and label to variable

                    tag = label+' '+acuracy+'%' #concatenate label and acurracy


                    #now draw te box
                    color = (225,0,0)
                    stroke = 2
                    end_corner_x = x + w
                    end_corner_y = y + h
                    
                    if float(acuracy) > 75: #show only the best results
                        cv2.putText(frame, tag , (end_corner_x, y+20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 1)#draw the text in the image

                    cv2.rectangle(frame, (x,y),(end_corner_x, end_corner_y),color,stroke) # draw the rect

            see_boxes()



            cv2.imshow('frame',frame)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyWindow()
     


PREDICT_CAM()


