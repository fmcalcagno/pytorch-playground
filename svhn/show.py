import argparse
import os
import time

import misc
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import dataset_digits

import model
from IPython import embed
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms


parser = argparse.ArgumentParser(description='PyTorch SVHN Example')
parser.add_argument('--channel', type=int, default=32, help='first conv channel (default: 32)')
parser.add_argument('--wd', type=float, default=0.001, help='weight decay')
parser.add_argument('--batch_size', type=int, default= 1, help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 1e-3)')
parser.add_argument('--gpu', default=1, help='index of gpus to use')
parser.add_argument('--ngpu', type=int, default=1, help='number of gpus to use')
parser.add_argument('--seed', type=int, default=117, help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=1,  help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=1,  help='how many epochs to wait before another test')
parser.add_argument('--logdir', default='log', help='folder to save to the log')
parser.add_argument('--data_root', default="C:\\Users\\fcalcagno\\Documents\\pytorch-playground_local\\svhn\\testingimages\\", help='folder to save the model')
parser.add_argument('--csv_path', default="C:\\Users\\fcalcagno\\Documents\\pytorch-playground_local\\svhn\\labels.csv", help='csv with the labels')
parser.add_argument('--decreasing_lr', default='80,120', help='decreasing strategy')
parser.add_argument('--use_pretrained',  default="Local", help='Use pretrained model or not')
parser.add_argument('--local_model',  default="C:\\Users\\fcalcagno\\Documents\\pytorch-playground_local\\svhn\\log\\best-90.pth", help='Where the local model is located')

args = parser.parse_args()
args.logdir = os.path.join(os.path.dirname(__file__), args.logdir)
misc.logger.init(args.logdir, 'train_log')

# select gpu
args.gpu = 1
args.ngpu = 1


args.cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# data loader and model
show_loader = dataset_digits.get(batch_size=args.batch_size, csv_path=args.csv_path, data_root=args.data_root, train=False, val=False, show=True)

toPIL=transforms.ToPILImage(mode="RGB")
try:
    for epoch in range(args.epochs):
        for batch_idx, (data, target) in enumerate(show_loader):
            im=toPIL(data.squeeze())
            #im=Image.fromarray(data.numpy())
            im.show()

except Exception as e:
    import traceback
    traceback.print_exc()