import numpy as np
import model
import os
import cv2
import glob
import dataset

import torch
from torchvision import  transforms
from torch.utils.data import DataLoader
from  torch.utils.data import Dataset
import torch.nn.functional as F

from scipy import ndimage
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.metrics import average_precision_score,classification_report


use_cuda = torch.cuda.is_available()
folder_data = "C:\\Users\\fcalcagno\\Documents\\pytorch-playground\\svhn\\data"

model_svhn = model.svhn(32,pretrained=True)

folder_input = "C:\\Users\\fcalcagno\\Documents\\pytorch-playground\\svhn\\testingimages"

images = [(cv2.imread(file),file) for file in glob.glob("C:\\Users\\fcalcagno\\Documents\\pytorch-playground\\svhn\\testingimages\\*.png")]


class MyDataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.data = [(cv2.imread(file),file) for file in glob.glob("C:\\Users\\fcalcagno\\Documents\\pytorch-playground\\svhn\\testingimages\\*.png")]
        #self.target = y
        self.transform = transform
        
    def __getitem__(self, index):
        img1=self.data[index][0]
        img1 = cv2.resize(img1, (32, 32)) 
        filename=os.path.basename(self.data[index][1])
        # Normalize your data here
        if self.transform:
            img1 = self.transform(img1)

        return img1,filename
    
    def __len__(self):
        return len(self.data)

transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
                ])

datasettest= MyDataset(transform)
val_loader= torch.utils.data.DataLoader (datasettest,batch_size=1)

model_svhn.eval()
device = torch.device("cuda" if use_cuda else "cpu")

model = model_svhn.to(device)

predictions=[]
reality=[]
torch.set_printoptions(precision=3,profile="short")

print ("Image Name, Reality, Prediction , Probabilities (1,2,3,4,5,6,7,8,9,0)")
for batch_id,(im,filename) in enumerate(val_loader):

    with torch.no_grad():
        
        im=im.to(device)    

        result= model_svhn(im)
        pred = result.max(1, keepdim=True)[1] +1
        if pred == 10: pred=pred-10
        file=filename[0]
        real=file[:-4]
        real=int(real[-1:])
        predictions.append(int(pred.cpu()))
        reality.append(real)
        soft=F.softmax(result, dim=1)
        
        print("Image Name: {}, Reality:{}, Prediction:{}, Probabilities:{}".format(filename,real,pred.cpu().data.numpy()[0,0],soft.data[0]))

print("Predictions: {}, Reality:{}".format(predictions,reality))


#target_names = ['0', '1', '2','3','4','5','6','7','8','9']
#target_names = ['0', '1']
#print(classification_report(reality, predictions, target_names=target_names))