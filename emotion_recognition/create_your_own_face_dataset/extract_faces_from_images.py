import numpy as np 
import cv2
import os 

#pretrain model of open cv that detect faces
face_cascade = cv2.CascadeClassifier('../cascade/haarcascade_frontalface_default.xml')

#a list of availables files on input images directory
file_list = os.listdir('input_images')

#will ise this to count iterations use the result number as a name for each image
n = 0


for i in file_list:
    n+=1
    
    #load image 1 by 1 as numpy array
    image = cv2.imread(('input_images/{}'.format(i)))
    
    #convert the image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #detect the face on my current image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for(x,y,w,h) in faces:
            
            print('face detected: {}'.format(n))
            
            #store only the face
            ROI_color = gray[y:y+h, x:x+w]
            
            #the path where i what to store my image and the name
            img_item = "output_images/7_IMG_{}.png".format(n)
            
            #save the image
            cv2.imwrite(img_item,ROI_color)