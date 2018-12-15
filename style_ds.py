import os
import random
from io import BytesIO as Bytes2Data
from os.path import join

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


def get_train_transform(size=(224, 224)):
    return transforms.Compose([
        transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

def get_test_transform(size=(224, 224)):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])   

class AEDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        self.data_dir = data_dir
        self.styles = os.listdir(data_dir)
        self.data = []

        for s in self.styles:
            style_folder = join(self.data_dir, s)
            img_names = os.listdir(style_folder)
            for img_name in img_names:
                self.data.append(self.get_bytes(join(style_folder, img_name)))

    def get_bytes(self, path):
        with open(path, 'rb') as f:
            return f.read()

    def __getitem__(self, index):
        img = Image.open(Bytes2Data(self.data[index])).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.data)


class ExtractGANDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        self.data_dir = data_dir
        self.styles = os.listdir(data_dir)
        self.style_id_set = set(range(len(self.styles)))
        self.data = []
        self.data_style_id = []
        self.quaternions = []

        for i, s in enumerate(self.styles):
            style_folder = join(self.data_dir, s)
            img_names = os.listdir(style_folder)
            style_data_bytes = []
            for img_name in img_names:
                style_data_bytes.append(self.get_bytes(join(style_folder, img_name)))
                self.data_style_id.append(i)
            self.data.append(style_data_bytes)
        
        self.data_flatten = sum(self.data, [])


    def get_bytes(self, path):
        with open(path, 'rb') as f:
            return f.read()


    def __getitem__(self, index):
        ori_img = self.data_flatten[index]
        source_style_id = self.data_style_id[index]
        target_style_id = random.choice(list(self.style_id_set - {source_style_id}))

        style_img, style_ref_img = random.choices(self.data[target_style_id], k=2)
        style_ori_img = random.choice(self.data[source_style_id])
        
        img_quaternion = [ori_img, style_img, style_ref_img, style_ori_img]

        img_quaternion = [Image.open(Bytes2Data(img)).convert('RGB') for img in img_quaternion]
        
        if self.transform is not None:
            img_quaternion = [self.transform(img) for img in img_quaternion]
        return img_quaternion

    def __len__(self):
        return len(self.data_flatten)
