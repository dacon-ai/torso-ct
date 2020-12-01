import cv2
import collections
import time 
from tqdm import tqdm
from PIL import Image
from functools import partial
train_on_gpu = True

import copy
import torch
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nibabel as nib
import matplotlib.pyplot as plt
import cv2
import pydicom
import time
import sys
from IPython.display import clear_output

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import TensorDataset, DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

import albumentations as albu
from albumentations import Compose,Resize,OneOf,RandomBrightness,RandomContrast,Normalize,HorizontalFlip,Blur,ElasticTransform,GridDistortion,OpticalDistortion,GaussNoise 
from albumentations.pytorch import ToTensor

from catalyst.data import Augmentor
from catalyst.dl import utils
from catalyst.data.reader import ScalarReader, ReaderCompose, LambdaReader#ImageReader
from catalyst.dl.runner import SupervisedRunner
#from catalyst.contrib.models.segmentation import Unet
from catalyst.dl.callbacks import DiceCallback, EarlyStoppingCallback, InferCallback, CheckpointCallback
from catalyst.dl.callbacks import JaccardCallback,PrecisionRecallF1ScoreCallback
import segmentation_models_pytorch as smp
import os

#os.environ["CUDA_VISIBLE_DEVICES"]="0"

seed = 1015
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASSES = 4

path = './dataset/data/'
original_path = '00002057_s1_Anonymized_00002057_converted_250_'
object_path = 'nii/00002057_s2_Anonymized_Segmented__00002057_converted_250.nii'
 
file_size = 2

train_meta = pd.DataFrame(
                        {
                        # 이후 데이터가 추가될 경우를 생각하여 10번 반복하였습니다.
                        # 실제로는 N개의 샘플이 추가된다면 N개의 row를 갖는 mata 데이터가 생성됩니다.
                         'original' :  [path+ '원본/'      + original_path for i in range(file_size)],
                         'musle'    :  [path+ '근육/'      + object_path for i in range(file_size)],
                         'fat'      :  [path+ '피하지방/'  + object_path for i in range(file_size)],
                         'innerfat' :  [path+ '내장지방/'  + object_path for i in range(file_size)],
                         'part'     :  [(path,object_path)for i in range(file_size)]
                        }            
                    )

test_meta = pd.DataFrame(
                        {
                         'original' :  [path+ '원본/'      + original_path for i in range(2)],
                         'musle'    :  [path+ '근육/'      + object_path for i in range(2)],
                         'fat'      :  [path+ '피하지방/'  + object_path for i in range(2)],
                         'innerfat' :  [path+ '내장지방/'  + object_path for i in range(2)],
                          'part'     :  [(path,object_path)for i in range(2)]
                        }            
                    )


class CT_Dataset(torch.utils.data.Dataset):
    def __init__(self, meta_data, 
                 transform=None,
                 preprocessing=None,
                 classes=None, 
                 augmentation=None, 
                ):
        self.meta_data = meta_data
        self.transforms = transform
        self.classes = classes
        self.preprocessing = preprocessing
        self.augmentation = augmentation
        self.imgsize = 256
        self.resize =  Compose([
                                  albu.Resize(height = self.imgsize, width = self.imgsize),
                               ])
        self.to_tensor = Compose([
                                  albu.Lambda(image=  to_tensor, mask=to_tensor),
                               ])
        
        
    def __getitem__(self,index):

        # Set index
        person_idx, img_index= divmod(index,40)
        img_index += 75

        # Read Data----------------------------------------
        # Read original - pydicom
        cts_path  = self.meta_data.loc[person_idx, 'original']
        ct = pydicom.dcmread(cts_path + str(img_index).zfill(3)).pixel_array
        ct = ct[70:-70,70:-70]
        image = ct / ct.max() * 255 
        image = Image.fromarray(image)
        image = np.array(image.convert('L'))[...,np.newaxis]

        # Read mask - nii
        ct_masks = []
        for body in ['근육/', '피하지방/','내장지방/']:
            mask_path = self.meta_data.loc[person_idx, 'part'][0] + body + self.meta_data.loc[person_idx, 'part'][1]
            ct_mask = nib.load(mask_path)
            ct_mask = np.array(ct_mask.get_fdata()[70:-70, 70:-70,img_index,:]) # aixs 3가 1개 존재 0번으로 인덱싱  
            ct_masks.append(ct_mask)
        target_mask = np.concatenate(ct_masks, axis = 2)
        # nii파일은 dicom파일과 다르게 transpose되어 있음.
        # 2번축은 그대로 두고 0번 축과 1번 축을 traspose
        target_mask = target_mask.transpose([1,0,2])
       
        
        #append backgraound chennel
        background = target_mask.sum(axis = 2) #(128, 128, 채널(musle, fat, innerfat))
        background = np.where(background != 0, 0, 1)[...,np.newaxis]
        target_mask = np.concatenate([target_mask, background], axis = 2)


        # Preprocessing----------------------------------------
        # Resize -> augmentation -> preprocessing(pretrained 모델) -> to tensor
        # resize (모두 적용)
        sample = self.resize(image=image, mask=target_mask)
        image, target_mask = sample['image'], sample['mask']

        #apply augmentation
        if self.augmentation:
            sample = self.augmentation(image=image, mask=target_mask)
            image, target_mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=target_mask)
            image, target_mask = sample['image'], sample['mask']

        # reshape for converting to tensor (모두 적용)
        sample = self.to_tensor(image=image, mask=target_mask)
        image, target_mask = sample['image'], sample['mask']
        
        return image, target_mask
    
    def __len__(self):
        return len(self.meta_data) * 40
    
    
def get_training_augmentation():
    transform = [
        albu.Transpose(p=0.5),
        albu.RandomRotate90(3),
        albu.Rotate(p=1),
        ]
    return albu.Compose(transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image = preprocessing_fn),
    ]
    return albu.Compose(_transform)


ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
DEVICE = 'cuda'

ACTIVATION = 'softmax'

model = smp.Unet(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    in_channels = 1,
    classes=CLASSES, 
    activation = ACTIVATION
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

train_dataset = CT_Dataset(
    train_meta,
    augmentation=get_training_augmentation(), 
    preprocessing=None, #get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

valid_dataset = CT_Dataset(
    test_meta,
    augmentation = None,
    preprocessing= None, #get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

BATCH_SIZE = 16

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, persistent_workers = True)
valid_loader = DataLoader(valid_dataset,  batch_size=40, shuffle=False, num_workers=2)
loaders = {
    "train": train_loader,
    "valid": valid_loader
}


num_epochs = 1
optimizer = torch.optim.Adam([
    {'params': model.decoder.parameters(), 'lr': 1e-2}, 
    {'params': model.encoder.parameters(), 'lr': 1e-3},  
])

scheduler = ReduceLROnPlateau(optimizer, factor=0.15, patience=2)
criterion = smp.utils.losses.DiceLoss(eps=1.)
runner = SupervisedRunner(device=device)

runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=loaders,
    callbacks=[
               DiceCallback(),
               JaccardCallback(),
               #EarlyStoppingCallback(patience=5, min_delta=0.001),
               ],
    # logdir=logdir,
    num_epochs=num_epochs,
    verbose=True
)