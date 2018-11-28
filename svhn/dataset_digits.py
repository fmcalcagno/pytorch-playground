import torch
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import os
from PIL import Image
import pandas as pd
import numpy as np
import cv2

class MyDigitsDataset(Dataset):
    def __init__(self, csv_path,img_path, transform=None):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: Optional transform to be applied on a sample.
        """
        self.data = pd.read_csv(csv_path,sep=";")
        self.name_files = np.asarray(self.data.iloc[:, 0])
        self.labels = np.asarray(self.data.iloc[:, 1])
        self.img_path = img_path
        self.transform = transform
        
    def __getitem__(self, index):
        # stuff
        single_image_label = self.labels[index] -1
        if single_image_label==-1:
            single_image_label=9

        img_name = os.path.join(self.img_path,self.name_files[index])
        data = Image.open(img_name)
        #data = data.resize((32, 32)) 
        #data=cv2.imread(img_name)
        #data = cv2.resize(data, (32, 32)) 
        #data = np.asarray(img)
        #data = np.transpose(data,(2,0,1))
        if self.transform is not None:
            data = self.transform(data)
        # If the transform variable is not empty
        # then it applies the operations in the transforms with the order that it is created.
        return (data, single_image_label)

    def __len__(self):
        return len(self.labels) 

def get(batch_size, csv_path='', data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    #data_root = os.path.expanduser(os.path.join(data_root, 'svhn-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building SVHN data loader with {} workers".format(num_workers))

    def target_transform(target):
        return int(target[0]) - 1

    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            MyDigitsDataset(
                csv_path=csv_path, img_path=data_root,
                transform=transforms.Compose([
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                    transforms.RandomAffine(10,scale=(0.08,1),shear=10),
                    transforms.Resize((32,32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
            ),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            MyDigitsDataset(
                csv_path=csv_path, img_path=data_root,
                transform=transforms.Compose([
                    transforms.Resize((32,32), interpolation=2),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
            ),      
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

