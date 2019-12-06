import time
import torch
import pickle
import torchvision.transforms as t
import torch.utils.data as data
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import os
import h5py
import random
from shutil import copyfile
import io
import yaml

class TextLogger():
    def __init__(self, title, save_path, append=False):
        print(save_path)
        file_state = 'wb'
        if append:
            file_state = 'ab'
        self.file = open(save_path, file_state, 0)
        self.log(title)
    
    def log(self, strdata):
        outstr = strdata + '\n'
        outstr = outstr.encode("utf-8")
        self.file.write(outstr)

    def __del__(self):
        self.file.close()

train_transform_bach = t.Compose([
    t.Resize((896, 896)),
    #t.Resize((448, 448)),
    #t.Pad(32, padding_mode='reflect'),
    #t.Pad(64, padding_mode='reflect'),
    t.RandomHorizontalFlip(0.5),
    t.RandomRotation([0, 360]),
    t.RandomCrop(448),
    t.ColorJitter(
        hue= 0.4,
        saturation=0.4,
        brightness=0.4,
        contrast=0.4),
    t.ToTensor(),
    ])

test_transform_bach = t.Compose([
    t.Resize((896, 896)),
    #t.Resize((448, 448)),
    t.CenterCrop(448),
    t.ToTensor()
    #t.FiveCrop(448),
    #t.ToTensor(),
    ])


class Bach_dataset(data.Dataset):
    def __init__(self, mode):
        
        self.mode = mode

        source = '/mnt/datasets/bach/'
        target =  '/scratch/ssd/eteh/'
        data_path = 'bach_%s.h5' % mode

        if not os.path.exists(target):
            os.makedirs(target)

        def copy_file(source, target, file_path):
            if not os.path.exists(os.path.join(target, file_path)):
                print('copyfing file:', file_path)
                copyfile(os.path.join(source, file_path), os.path.join(target,file_path))

        copy_file(source, target, data_path)

        self.ys = []
        self.I = []
        self.h5_file = os.path.join(target, data_path)

        dat = h5py.File(self.h5_file, 'r')
        if 'test' not in self.mode:
            all_ys_labels = torch.Tensor(dat['y']).squeeze().long()

        if 'train' in self.mode:
            self.transform = train_transform_bach
        else:
            self.transform = test_transform_bach

        if 'test' not in self.mode:
            for ix in range(len(all_ys_labels)):
                self.ys += [all_ys_labels[ix].item()]
                self.I += [ix]
        else:
            len_t = len(dat['x'])
            self.ys = torch.ones(len_t) * -1
            self.I = list(range(len_t))

        pil2tensor = t.ToTensor()

        if not os.path.exists('data/mean_std_bach.pt'):
            mean_std = {}
            mean_std['mean'] = [0,0,0]
            mean_std['std'] = [0,0,0]

            print('Calculating mean and std')
            for ix in tqdm(range(len(all_ys_labels))):
                img = pil2tensor(Image.open(io.BytesIO(dat['x'][ix])))
                for cix in range(3):
                    mean_std['mean'][cix] += img[cix,:,:].mean()
                    mean_std['std'][cix] += img[cix,:,:].std()

            for cix in range(3):
                mean_std['mean'][cix] /= len(all_ys_labels)
                mean_std['std'][cix] /= len(all_ys_labels)

            torch.save(mean_std, 'data/mean_std_bach.pt')
        else:
            mean_std = torch.load('data/mean_std_bach.pt')

        #normalize = t.Normalize(mean=mean_std['mean'], std=mean_std['std'])
        #normalize = t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        normalize = t.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.transform.transforms.append(normalize)
        '''
        if 'train' in self.mode:
            self.transform.transforms.append(normalize)
        elif 'val' in self.mode or 'test' in self.mode:
            self.transform.transforms.append(t.Lambda(lambda crops: torch.stack([t.ToTensor()(crop) for crop in crops])))
            self.transform.transforms.append(t.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        '''
        self.data = None
        dat.close()

    def __getitem__(self, index):
        if self.data == None:
            self.data = h5py.File(self.h5_file, 'r')
        curr_index = self.I[index]
        img = Image.open(io.BytesIO(self.data['x'][curr_index]))
        if self.mode != 'test':
            target = self.ys[index]
        img = img.convert("RGB")

        img = self.transform(img)
        #print(self.transform)
        if self.mode != 'test':
            return img, target
        else:
            return img, self.data['f'][curr_index].decode('UTF-8')

    def __len__(self):
        return len(self.ys)


    def nb_classes(self):
        return len(set(self.ys))

