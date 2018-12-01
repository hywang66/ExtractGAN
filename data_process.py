import os
from PIL import Image
import torchvision
from torchvision import transforms 
from torch.utils import data
import numpy as np
import random
class Dataset(data.Dataset):
  def __init__(self, style_list):
        self.style_list= style_list
  def __len__(self):
        return int(min([len(style) for style in self.style_list])/2)
  def __getitem__(self, index):
        num_list = random.sample(range(0,4),2)
        return data_process(self.style_list[num_list[0]],self.style_list[num_list[1]])

def return_transform(size):
    transform = transforms.Compose([
          transforms.Resize(size),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])
    return transform

style_transform = return_transform(224)
#normal_transform = return_transform(64)

def data_process(A,B):
    a = list(np.random.randint(0,len(A),size = 2))
    b = list(np.random.randint(0,len(B),size = 2))
    return tuple((style_transform(A[a[0]]),style_transform(B[b[0]]),style_transform(A[a[1]]),style_transform(B[b[1]])))


def loading(dir1):
    dir1 = os.path.abspath(os.path.join(os.pardir,"ExtractGAN/data/"+dir1))
    dataset = torchvision.datasets.ImageFolder(root=dir1+"/")
    A_test = []
    B_test = []
    A_train = []
    B_train =[]
    for i in range(len(dataset)):
        x,y = dataset[i]
        if y ==0:
            A_test.append(x)
        elif y ==1:
            B_test.append(x)
        elif y ==2:
            A_train.append(x)
        else:
            B_train.append(x)
    return [A_test,B_test],[A_train,B_train]

def data_loader(dir1,dir2):
    test_list =[]
    train_list = []
    temp_test,temp_train = loading(dir1)
    test_list+=temp_test
    train_list+=temp_train
    temp_test,temp_train = loading(dir2)
    test_list+=temp_test
    train_list+=temp_train
    train_dataset = Dataset(train_list)
    test_dataset = Dataset(test_list)
    
    return train_dataset,test_dataset

