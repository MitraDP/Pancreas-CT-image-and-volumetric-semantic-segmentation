# -*- coding: utf-8 -*-
"""PancreasCT_train_test_2D_3D_bck_2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tGjZ1Agbt2gnhgSj9hDn9LUtStXfORSg

#Pancreas Cancer

##Install the NVIDIA System Management Interface
"""

!ls -sf /opt/bin/nvidia-smi /usr/bin/nvidia-smi
!nvidia-smi
!nvcc --version

"""##Library imports and the environment setting"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install pydicom
# !pip install pillow
# !pip install torchio
# !pip install torch-lr-finder
# 
# import os
# import glob
# import numpy as np
# import pandas as pd
# import matplotlib
# import matplotlib.pyplot as plt
# from PIL import Image
# import cv2
# from collections import OrderedDict
# import random
# from random import shuffle
# 
# import pydicom as dicomio
# import nibabel as nib
# 
# import torch
# import torch.utils.data
# import torchvision
# from torch.utils.data import Dataset
# from torchsummary import summary
# import torch.optim as optim
# import torchvision.transforms as transforms
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from torch_lr_finder import LRFinder
# import albumentations as A
# import torchio as tio

#check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available. Training on CPU ...')
else:
    print('CUDA is available. Training on GPU ...')

##Set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

set_seed(51)

from loss import TverskyLoss
from net import UNet_2D, UNet_3D
from volume_patch_composer import volume_composer, patch_creator
from dataset import Pancreas_2D_dataset, Pancreas_3D_dataset, partitioning
from metrics import performance_metrics
from train import train_2D, train_3D
from inference import get_inference_performance_metrics_2D
from inference import get_inference_performance_metrics_3D
from inference import  visualize_patient_prediction_2D
from inference import visualize_patient_prediction_3D
from inference import volume

"""##Import the pancreas datasets"""

from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# #make a directory for the original data
# !mkdir data/
# #make a directory for the resized 3D data
# !mkdir data3D
# #upload CT zip file
# !cp /content/drive/MyDrive/Pancreas-CT.zip /content/
# !unzip   Pancreas-CT.zip
# !rm Pancreas-CT.zip
# !rm -r sample_data
#

"""
For each patient create 2 folders in the "data" directory one for the CT and
one for the mask.
"""
dir_list = []
for i in range(1, 83):
    patient_label = '{:04d}'.format(i)
    pth = os.path.join('data', 'Patient' + patient_label)
    dir_list.append(pth)   
for dir in dir_list:
    p = dir +'/Masks'
    os.makedirs(p)
    p = dir +'/CT'
    os.makedirs(p)

"""###Load CT (DICOM) and mask (NIfTI) files and save them as png"""

#The dataset has 82 patient ID/label, e.g. 0057
#Upload each patient's annotation folder
for i in range(1,83):
  patient_label = '{:04d}'.format(i)
  pth = os.path.join('/content', 'drive', 'MyDrive', 'Masks', 'label'+ 
                     patient_label +'.nii.gz')
  img = nib.load(pth)
  img_data = img.get_fdata()
  #load and save patient's annotation slices
  for s in range (img_data.shape[2]):
    slice_label = '{:03d}'.format(s+1)
    slice_img = img_data[:, :, s]
    slice_path = os.path.join('/content', 'data', 'Patient' + patient_label,
                              'Masks', "M_" + slice_label + '.png' )    
    cv2.imwrite(slice_path, slice_img)

#Read each patient's CT slices and save them as pixel arrays
#Since the gridsampler samples in C x W x H x D, transpose CT slices. 
#The dataset has 82 patient ID/label, e.g. 0057
for i in range(1,83):
  patient_label = '{:04d}'.format(i)
  g = glob.glob('/content/Pancreas-CT/PANCREAS_' + patient_label + '/*/*/*.dcm')
  #load and save patient's CT slices
  for i, f in enumerate(g):
    im_label = g[i].split('/')[-1].split('-')[1].split('.')[0]
    im_path  = os.path.join('/content', 'data', 'Patient' + patient_label, 
                            'CT', 'CT_'+ im_label + '.png' )
    cv2.imwrite(im_path,dicomio.read_file(g[i]).pixel_array.transpose(1,0))

#remove the original CT folder data to save memory
!rm -r Pancreas-CT

"""###Create path lists and examine patients data"""

patient_path_list = {} #A dictionary of patients CT and Masks paths
patient_path_list['CT'] = {} 
patient_path_list['Masks'] = {}
patient_image_cnt_CT = {} #A dictionary of the patient's number of CT slices 
patient_image_cnt_Mask = {} #A dictionary of the patient's number of Masks slices
#The dataset has 82 patient ID/label, e.g. 0057
for i in range(1,83):
  patient_label = '{:04d}'.format(i)
  patient_path_list ['CT']['Patient'+str(patient_label)] \
  = sorted(glob.glob('/content/data/Patient' + patient_label + '/CT/*.png'))        
  patient_image_cnt_CT['Patient'+str(patient_label)] \
  = len (patient_path_list ['CT']['Patient'+str(patient_label)])  
  patient_path_list ['Masks']['Patient'+str(patient_label)] \
  = sorted(glob.glob('/content/data/Patient' + patient_label + '/Masks/*.png'))   
  patient_image_cnt_Mask['Patient'+str(patient_label)] \
  = len (patient_path_list ['Masks']['Patient'+str(patient_label)])

""" 
Identify and remove patients with zero or inconsistent number of CT and mask 
slices.
"""
keys_to_delete = [k for k in patient_image_cnt_CT if patient_image_cnt_CT[k] \
                  != patient_image_cnt_Mask[k] or patient_image_cnt_CT[k]==0 \
                  or patient_image_cnt_Mask[k]==0 ]
for k in keys_to_delete:
    del patient_image_cnt_CT[k],patient_image_cnt_Mask[k],
    patient_path_list['CT'][k], patient_path_list['Masks'][k]

patient_cnt = len(patient_path_list['CT'].keys())  #number of patients left

#Number of slices per patient statistics
a = [*patient_image_cnt_Mask.values()]
print('max:', np.max(a), 'mean:', int(np.round(np.mean(a))), 'median:',
      int(np.median(a)), 'min:', np.min(a))

"""##Set the hyperparameters"""

#Define the type of segmentation (2D or 3D): bool
unet_2d = False  

#Volume resize parameters (d1,d2,d3) are (height, width,depth)
d1 = torch.linspace(-1, 1, 256)
d2 = torch.linspace(-1, 1, 256)
d3 = torch.linspace(-1,1, 128)

#Patch parameters for volumetric segmentation
if unet_2d == False:
  #kernel size
  kc, kh, kw = 32,64,64
  #stride  
  dc, dh, dw = 32,64,64

batch_size = 16
num_workers = 0

#Define type of optimizer as either 'Adam' or 'SGD'
optimizer_type = 'Adam' """adjust the learning rate in the
                           "Specify the loss function and optimizer" section"""

"""If you are willing to find the maximum learning rate using the One Cycle 
learning rate policy set lr_find to True"""
lr_find = False   
n_epochs = 1
inference_only = False #If you wish to use the pretrained model set to True

threshold = 0.5  # Threshold value to create binary image 

split_ratio = [0.70, 0.10, 0.20]   # A list of the (train,val,test) split ratio

"""##Volume resize"""

#Create a grid (d1,d2,d3) to be used for volume resizing
meshx, meshy, meshz = torch.meshgrid((d1, d2, d3))
grid = torch.stack((meshx, meshy, meshz), 3)
grid = grid.unsqueeze(0) # add batch dim

#Resize patients' CT and Masks using the same grid
for patient in patient_image_cnt_CT:
    volume_composer(patient, patient_image_cnt_CT, patient_path_list, grid)

"""###Compare the resized volume slices with their counterpart in the original dataset"""

#sample patient 
p = 'Patient0067'
#sample slice number in the resized volume
n = 60
#n_o is approximately the slice number in the original volume
n_o = str(int( n * patient_image_cnt_Mask[p] / d3.numpy().size))
im = torch.load('/content/data3D/' + p + '_CT.pt')
m = torch.load('/content/data3D/' + p + '_Mask.pt')
im = im.numpy()
m = m.numpy()
im = np.squeeze(im)[:,:,n]
m = np.squeeze(m)[:,:,n]
im_o = Image.open('/content/data/' + p + '/CT/CT_' + n_o + '.png')
m_o = Image.open('/content/data/' + p + '/Masks/M_' + n_o + '.png')
im_o_t = np.transpose(im_o)
m_o_t = np.transpose(m_o)
plt.figure(figsize=(15,10))
plt.subplot(2,3,1)
plt.imshow(im, cmap="gray", interpolation= None)
plt.title('resized CT')
plt.subplot(2,3,2)
plt.imshow(m, cmap="gray", interpolation= None)
plt.title('resized annotation mask')
plt.subplot(2,3,3)  
plt.imshow(im, cmap="gray", interpolation= None)
plt.imshow(m, cmap="jet", alpha = 0.3, interpolation= None)
plt.subplot(2,3,4)
plt.imshow(im_o_t, cmap="gray", interpolation= None)
plt.title('original CT transposed')
plt.subplot(2,3,5)
plt.imshow(m_o_t, cmap="gray", interpolation= None)
plt.title('original annotation mask transposed')
plt.subplot(2,3,6)  
plt.imshow(im_o_t, cmap="gray", interpolation= None)
plt.imshow(m_o_t, cmap="jet", alpha = 0.3, interpolation= None)

#remove "data" directory as we don't need it anymore
!rm -r data

"""###Create and save slices for 2D training based on the resized volumes"""

if unet_2d:
    !mkdir data/
    slice_cnt = d3.numpy().size
    ##Recreate a dictionary of patients CT and Masks paths
    patient_path_list = {}
    patient_path_list['CT'] = {}
    patient_path_list['Masks'] = {}

    for p in patient_image_cnt_CT.keys():
        path_CT_folder = os.path.join('data', p, 'CT')
        path_mask_folder = os.path.join('data', p, 'Masks')
        os.makedirs(path_CT_folder)
        os.makedirs(path_mask_folder)
        #load 3D CT image
        im =  torch.load('/content/data3D/' + p + '_CT.pt')
        #load 3D mask 
        m =  torch.load('/content/data3D/' + p + '_Mask.pt')
        ## Transform CT and mask to numpy array
        im = im.numpy().squeeze(0).squeeze(0)
        m = m.numpy().squeeze(0).squeeze(0)
        for s in range(slice_cnt):
            #create a 3digit label for each slice
            label = '{:03d}'.format(s)
            """
            save each patients CT and Mask in a designated folder, e.g. patient
            17, CT slice 100 would be '/content/data/Patient0017/CT/CT_100.png'
            """
            ct_path  = os.path.join('/content', 'data', p, 'CT', 'CT_'+ label +
                                    '.png' )
            mask_path = os.path.join('/content', 'data', p, 'Masks', "M_" + 
                                     label + '.png' )    
            cv2.imwrite(ct_path, im[:, :, s])
            cv2.imwrite(mask_path, m[:, :, s])     
        patient_path_list ['CT'][p] = sorted(glob.glob('/content/data/' + p +
                                                       '/CT/*.png'))        
        patient_path_list ['Masks'][p] = sorted(glob.glob('/content/data/' +
                                                          p + '/Masks/*.png'))

"""####Check the 2D png files"""

if unet_2d:
    #for patient 17,47,77 check the CT and mask slice number 100
    CT_0 = Image.open('/content/data/Patient0017/CT/CT_100.png')
    CT_1 = Image.open('/content/data/Patient0047/CT/CT_100.png')
    CT_2 = Image.open('/content/data/Patient0077/CT/CT_100.png')
    slice_0 = Image.open('/content/data/Patient0017/Masks/M_100.png')
    slice_1 = Image.open('/content/data/Patient0047/Masks/M_100.png')
    slice_2 = Image.open('/content/data/Patient0077/Masks/M_100.png')
    plt.figure(figsize=[15,15])
    plt.subplot(3,3,1)
    plt.imshow(CT_0, cmap="gray", interpolation= None)
    plt.subplot(3,3,2)
    plt.imshow(CT_1, cmap="gray", interpolation= None)
    plt.subplot(3,3,3)
    plt.imshow(CT_2, cmap="gray", interpolation= None)
    plt.subplot(3,3,4 )
    plt.imshow(slice_0, cmap="gray", interpolation= None)
    plt.subplot(3,3,5 )
    plt.imshow(slice_1, cmap="gray", interpolation= None)
    plt.subplot(3,3,6 )
    plt.imshow(slice_2, cmap="gray", interpolation= None)
    plt.subplot(3,3,7)
    plt.imshow(CT_0, cmap="gray", interpolation= None)
    plt.imshow(slice_0, cmap="jet", alpha =0.3, interpolation= None)
    plt.subplot(3,3,8)
    plt.imshow(CT_1, cmap="gray", interpolation= None)
    plt.imshow(slice_1, cmap="jet", alpha =0.3, interpolation= None)
    plt.subplot(3,3,9)
    plt.imshow(CT_2, cmap="gray", interpolation= None)
    plt.imshow(slice_2, cmap="jet", alpha =0.3, interpolation= None)

"""##Patients' ID partitioning"""

#stratify split patients into 3 sets: train, valid, test
part = partitioning([*patient_image_cnt_CT.keys()], split_ratio = [0.7,0.1,0.2])

"""###Data partitions"""

if unet_2d:
    """
    2D data partitioning: Create 3 partitions (train, valid,test) where each are
    dictionaries of CT and mask paths.
    """
    partition_train = {}
    partition_train ['CT'] = []
    partition_train ['Masks'] = []
    for p in part['train']:
        partition_train ['CT'].extend(patient_path_list ['CT'][p] )
        partition_train ['Masks'].extend(patient_path_list ['Masks'][p] )
    partition_valid = {}
    partition_valid ['CT'] = []
    partition_valid ['Masks'] = []
    for p in part['valid']:
        partition_valid ['CT'].extend(patient_path_list ['CT'][p] )
        partition_valid ['Masks'].extend(patient_path_list ['Masks'][p])        
    partition_test= {}
    for p in part['test']:
        partition_test [p] = {}  
        partition_test[p] ['CT'] = []
        partition_test[p] ['Masks'] = []
        partition_test[p] ['CT'].extend(patient_path_list ['CT'][p])
        partition_test[p] ['Masks'].extend(patient_path_list ['Masks'][p])
else:
    """
    Create subvolumes (patches) for each patient's CT and mask, and save the
    patches (torch tensors) in the corresponding dictionary, i.e. based on
    the patient's partition. The 'test' patches will be created seperately per 
    patient in the "Get the inference performance metrics" section.
    """
    CT_patches = {}
    mask_patches ={}
    for p in ['train', 'valid']:
        CT_patches[p], mask_patches[p] = patch_creator(part[p],
                                                       kw, kh, kc, dw, dh, dc)

"""## Constructing the dataset and the dataloader"""

# Construct the dataset
if unet_2d:
    dataset_train = Pancreas_2D_dataset (partition_train, augment= True)
    dataset_valid = Pancreas_2D_dataset (partition_valid, augment= False)        
    dataset_test ={}
    # The test partition is arranged per patient
    for p in partition_test:
      dataset_test[p] = Pancreas_2D_dataset (partition_test[p], augment = False)
else:
    #
    dataset_train = Pancreas_3D_dataset (CT_patches['train'], 
                                         mask_patches['train'], augment= True)
    dataset_valid = Pancreas_3D_dataset (CT_patches['valid'],
                                         mask_patches['valid'], augment= False)

"""
Generators (data loaders) for the train and valid sets. The test loader is 
in the "Generate predictions" section.
"""
loaders={}
loaders['train'] = torch.utils.data.DataLoader(dataset_train, 
                                               batch_size=batch_size, 
                                               shuffle=True, 
                                                num_workers=num_workers)
loaders['valid'] = torch.utils.data.DataLoader(dataset_valid, 
                                               batch_size=batch_size, 
                                               shuffle=False, 
                                               num_workers=num_workers)

"""###Get sample batch from loader"""

batch = iter(loaders['valid'])

image, mask = next(batch)

if unet_2d:
    for im, m in zip(image, mask):
        im = im.numpy()
        m = m.numpy()
        im = np.squeeze(im)
        m = np.squeeze(m)
        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow(im, cmap="gray", interpolation= None)
        plt.subplot(1,3,2)
        plt.imshow(m, cmap="gray", interpolation= None)
        plt.subplot(1,3,3)  
        plt.imshow(im, cmap="gray", interpolation= None)
        plt.imshow(m, cmap="jet", alpha = 0.3, interpolation= None)
else:
    for im, m in zip(image, mask):
        #transfer C x D x H x W to C x W x H x D 
        im = im.permute(0,3,2,1)
        m = m.permute(0,3,2,1)
        im = im.numpy()
        m = m.numpy()
        im = np.squeeze(im)
        m = np.squeeze(m)
        plt.figure(figsize=(8,8))
        plt.subplot(4,3,1)
        plt.imshow(im[:,:,20], cmap="gray", interpolation= None)
        plt.subplot(4,3,2)
        plt.imshow(m[:,:,20], cmap="gray", interpolation= None)
        plt.subplot(4,3,3)
        plt.imshow(im[:,:,20], cmap="gray", interpolation= None)
        plt.imshow(m[:,:,20], cmap="jet", alpha = 0.3, interpolation= None)

"""##Obtain Model Architecture"""

# instantiate the unet
if unet_2d:
    model = UNet_2D(1,1,32,0.2)
else:
    model = UNet_3D(1,1,32,0.2)

# if GPU is available, move the model to GPU
if train_on_gpu:
    model.cuda()

if unet_2d:
  summary(model, (1,256, 256), batch_size = batch_size)
else:
  summary(model, (1, 32, 64, 64), batch_size = batch_size)

"""##Specify the loss function and optimizer
 
"""

criterion = TverskyLoss(1e-8,0.3,.7)
#lr_find = False
# Optimizer
if optimizer_type == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
else:
    optimizer = optim.Adam(model.parameters(), lr = .005)

"""###Learning rate scheduler"""

"""
If lr_find is True, after running this cell, assign the scheduler's max_lr to 
the suggested maximum lr and then set lr_find to False in the "Set the parameters"
section. Set the lr in the optimizer 1/10 of max_lr. Then re_run the code. 
"""
if lr_find == False:
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.08, steps_per_epoch=len(loaders['train']), epochs=n_epochs)        #(optimizer, max_lr=0.01, total_steps=4000)
else:
    #https://github.com/davidtvs/pytorch-lr-finder
    desired_batch_size, real_batch_size = batch_size, batch_size
    accumulation_steps = desired_batch_size // real_batch_size
    lr_finder = LRFinder(model, optimizer, criterion, device='cuda')
    lr_finder.range_test(loaders['train'], end_lr=1, num_iter=100, step_mode='exp')
    lr_finder.plot() # to inspect the loss-learning rate graph
    lr_finder.reset() # to reset the model and optimizer to their initial state

"""##Train and validate the model"""

if inference_only == False:
  # train the model
  if unet_2d:
      model = train_2D(n_epochs, loaders, model, optimizer, criterion, 
                       train_on_gpu, performance_metrics, 'model.pt', threshold)
  else:
      model = train_3D(n_epochs, loaders, model, optimizer, criterion, 
                       train_on_gpu, performance_metrics, 'model.pt', threshold)
else:
  # load the model that got the best validation accuracy or a trained model
  model.load_state_dict(torch.load('model.pt'))

# plot the variation of train and validation losses vs n_epochs
loss=pd.read_csv('performance_metrics.csv',header=0,index_col=False)
plt.plot(loss['epoch'], loss['Training Loss'], 'r', loss['epoch'],
         loss['Validation Loss'],'g')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.legend(labels=['Train','Valid'])
plt.show()

# plot the generalization error vs n_epochs
plt.plot(loss['epoch'],loss['Training Loss']-loss['Validation Loss'])
plt.xlabel('epochs')
plt.ylabel('Generalization Error')
plt.show()

"""##Generate predictions

###Get the inference performance metrics
"""

if unet_2d:
    df =get_inference_performance_metrics_2D(model, part['test'], dataset_test, batch_size, train_on_gpu, threshold)
else:
    df = get_inference_performance_metrics_3D(model, part['test'], Pancreas_3D_dataset, batch_size, train_on_gpu, threshold, kw, kh, kc, dw, dh, dc)

#metrics per patient
df

#The inference performance metrics stats
df.describe()

"""###Visualize the inference results"""

#sample patient 57
patient = 'Patient0057'
if unet_2d:
    visualize_patient_prediction_2D(model, patient, dataset_test, batch_size, 
                                    train_on_gpu, threshold) 
else: 
    visualize_patient_prediction_3D(model, patient, Pancreas_3D_dataset, 
                                    batch_size, train_on_gpu, threshold,
                                    kw, kh, kc, dw, dh, dc)

!pip install pipreqs
!pipreqs