# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 20:28:24 2020

@author: Administrator
"""


import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as Data
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.models as models
from sklearn.metrics import precision_score, recall_score, f1_score

root_3="train/new_picture/"
root_4="validation/new_picture/"

root_5="test/new_picture/"

EPOCH = 100              # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE =64
LR = 0.01            # learning rate
BATCH_SIZE1=16

transform_train=transforms.Compose([
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor()])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(1)

def default_loader(path):
    return Image.open(path).convert('L')


class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
            
        return img,label

    def __len__(self):
        return len(self.imgs)
    
    
train_data=MyDataset(txt=root_3+'train.txt', transform=transform_train)

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


validation_data=MyDataset(txt=root_4+'validation.txt', transform=transform_train)

validation_loader = Data.DataLoader(dataset=validation_data, batch_size=BATCH_SIZE1, shuffle=True)



test_data=MyDataset(txt=root_5+'test.txt', transform=transform_train)

test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE1, shuffle=True)




class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 16, 16)
                nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=3,              # filter size
                stride=1,                   # filter movement/step
                padding=1,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 16, 16)
                nn.ReLU(),                      # activation
                nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 8, 8)
        )
        
        self.conv2 = nn.Sequential(         # input shape (16, 8, 8)
                nn.Conv2d(16, 32, 3, 1, 1),     # output shape (32, 8, 8)
                nn.ReLU(),                      # activation
                nn.MaxPool2d(2),                # output shape (32, 4, 4)
        )
        
        
        self.conv3 = nn.Sequential(         # input shape (32, 4, 4)
                nn.Conv2d(32, 64, 3, 1, 1),     # output shape (64, 4, 4)
                nn.ReLU(),                      # activation
                nn.MaxPool2d(4),                # output shape (64, 1, 1)
        )
        
       
       
        self.out = nn.Linear(64 * 1 * 1, 2)
        
       
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x    # return x for visualization



cnn = CNN()


print(cnn)  # net architecture

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted


for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
        output = cnn(b_x)[0]               # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

    print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())
    
    correct = 0
    total = 0
    
    y_pred=[]
    y_true=[]
    
    
    for data in validation_loader:
        data = data.to(device)
        outputs = cnn(data)
        _, predicted = torch.max(outputs, 1)
        for j in range(len(predicted)):
            y_pred.append(predicted.cpu().tolist()[j])
            y_true.append(data.t.cpu().tolist()[j])
            total += data.t.size(0)
                
            correct += (predicted == data.t).sum().cpu().item()

    p=precision_score(y_true, y_pred,average='binary')
    r=recall_score(y_true, y_pred,average='binary')
    f1=f1_score(y_true, y_pred,average='binary')
    acc=correct/total
    print('p:',p,'r:',r,'f1',f1,'acc',acc)

    
    correct = 0
    total = 0
    
    y_pred=[]
    y_true=[]


    for data in test_loader:
        data = data.to(device)
        outputs = cnn(data)
        _, predicted = torch.max(outputs, 1)
        for j in range(len(predicted)):
            y_pred.append(predicted.cpu().tolist()[j])
            y_true.append(data.t.cpu().tolist()[j])
            total += data.t.size(0)
                
            correct += (predicted == data.t).sum().cpu().item()

    p=precision_score(y_true, y_pred,average='binary')
    r=recall_score(y_true, y_pred,average='binary')
    f1=f1_score(y_true, y_pred,average='binary')
    acc=correct/total
    print('p:',p,'r:',r,'f1',f1,'acc',acc)
    
    

    
    if loss<0.002:
        torch.save(cnn,'cnn.pkl')
        break
    


    
    
    
    
    
    
    
    
    
