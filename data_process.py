import os
from PIL import Image
import torchvision
from torchvision import transforms 
from random import shuffle
from torch.utils import data
import numpy as np

class Dataset(data.Dataset):
  def __init__(self, A,B):
        self.A = A
        self.B = B
  def __len__(self):
        return int(min(len(self.A),len(self.B))/2)
  def __getitem__(self, index):
        return data_process(self.A,self.B)

def return_transform(size):
    transform = transforms.Compose([
          transforms.Resize(size),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])
    return transform

style_transform = return_transform(224)
normal_transform = return_transform(64)

def data_process(A,B):
    a = list(np.random.randint(0,len(A),size = 2))
    b = list(np.random.randint(0,len(B),size = 2))
    return tuple((normal_transform(A[a[0]]),normal_transform(B[b[0]]),style_transform(A[a[1]]),style_transform(B[b[1]])))

def data_loader(dir1):
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
    train = data.DataLoader(Dataset(A_train,B_train),batch_size=32, shuffle=True)
    test = data.DataLoader(Dataset(A_test,B_test),batch_size=32, shuffle=True)
    return train,test
