#-----------------------------------------------------------------------#
#                          Library imports                              #
#-----------------------------------------------------------------------#
import torch
import torch.utils.data
from torch.utils.data import Dataset
import albumentations as A
import torchio as tio
import random
from random import shuffle


#-----------------------------------------------------------------------#
#                             partitioning                              #
#       Splits the patient into 3 groups: train, val, and test      #
#-----------------------------------------------------------------------#
# Returns a dictionary with 3 keys ['train', 'val', 'test'], where the  #
# values are lists of patient ids.                                      #
# The ids are stratified shuffled split.                                #
#-----------------------------------------------------------------------#
# split_ratio:      A list of the (train,val,test) split ratio,         #
#                   e.g. [0.7, 0.1, 0.2].                               #
# partition:        A dictionary of train, valid, test patient IDs.     #
# l:                Total number of patients.                           #
# split_pt:         split the length of patients list using             #
#                   the split_ratio.                                    #
#-----------------------------------------------------------------------#
def partitioning(patients_list, split_ratio):
    part = {'train':[], 'valid':[], 'test':[]}
    #Shuffle the patient list
    random.shuffle(patients_list)
    l = len(patients_list)
    # find the split indices
    split_pt = [int(split_ratio[0]*l), int((split_ratio[0]+\
                                            split_ratio[1])*l)]
    # stratify split the paths
    part['train'] =  patients_list [:split_pt[0]]
    part['valid'] =  patients_list [split_pt[0]: split_pt[1]]
    part['test'] =  patients_list [split_pt[1]:]
    print('train: ', len(part['train']),' ','valid: ', 
          len(part['valid']),' ','test: ', len(part['test']),' ',
          'total: ',len(part['train'])+len(part['valid'])+len(part['test']))
    #Return the partitions
    return part


#-----------------------------------------------------------------------#
#                        Pancreas_2D_dataset                            #
#                   constructs dataset for 2D model                     #
#-----------------------------------------------------------------------#
# References:                                                           #
# https://github.com/albumentations-team/albumentations                 #
# https://pytorch.org/docs/stable/torchvision/transforms.html           #
# https://github.com/pytorch/vision/blob/master/references/             #
#          segmentation/transforms.py                                   #
#-----------------------------------------------------------------------#
# returns: a dataset of 2D image and mask pairs
#-----------------------------------------------------------------------#
# partition:    either train, valid, or test partition                  #
# augment:      True if augmentation is required                        #
# aug:          augmentation pipeline                                   #
#-----------------------------------------------------------------------#
class Pancreas_2D_dataset(Dataset):    
    def __init__(self,partition, augment = False):
        self.partition = partition
        self.augment = augment
        
        self.aug = A.Compose([
        A.OneOf([
            A.Rotate(limit=15),
            A.VerticalFlip()], p=0.7),
        A.PadIfNeeded(min_height=256, min_width=256, p=1),
        A.GridDistortion(p=0.5)])
    
    def __len__(self):
        return (len(self.partition['CT']))
    
    def __getitem__(self, idx):
        # Generate one batch of data
        image = Image.open(self.partition['CT'][idx]) 
        mask = Image.open(self.partition['Masks'][idx])
        image = np.array(image)
        mask = np.array(mask)
        t1 = transforms.ToTensor()
        if self.augment:
            augmented = self.aug(image=image, mask=mask)        
            image = t1(augmented['image'])            
            """
            https://pytorch.org/docs/stable/torchvision/transforms.html
            Because the input image is scaled to [0.0, 1.0], totensor 
            transformation should not be used when transforming target 
            image masks.
            https://github.com/pytorch/vision/blob/master/references/segmentation/\
            transforms.py
            """
            mask = torch.as_tensor(augmented['mask']) 
        else:
            image = t1(image)
            mask = torch.as_tensor(mask)

        mask = torch.unsqueeze(mask,0)
        image = image.type(torch.FloatTensor)
        # note that mask is integer
        mask = mask.type(torch.IntTensor)
       
        # Return image and mask pair tensors
        return image, mask

#-----------------------------------------------------------------------#
#                        Pancreas_3D_dataset                            #
#                   constructs dataset for 3D model                     #
#-----------------------------------------------------------------------#
# Reference:                                                            #
# https://github.com/fepegar/torchio                                    #
# https://pytorch.org/docs/stable/torchvision/transforms.html           #
#-----------------------------------------------------------------------#
# returns: a dataset of 3D image and mask pairs                         #
#-----------------------------------------------------------------------#
# CT_partition:   either train, valid, or test partition of CT patches  #
# mask_partition: either train, valid, or test partition of mask patches#
# augment:        True if augmentation is required                      #
# aug:            augmentation pipeline                                 #
#-----------------------------------------------------------------------#    
class Pancreas_3D_dataset(Dataset):    
    def __init__(self, CT_partition, mask_partition, augment = False):
        self.CT_partition = CT_partition
        self.mask_partition = mask_partition
        self.augment = augment
       
    def __len__(self):
        return (len(self.CT_partition))
    
    def __getitem__(self, idx):    
        # Generate one batch of data
        # ScalarImage expect 4DTensor, so add a singleton dimension
        image = self.CT_partition[idx].unsqueeze(0)
        mask = self.mask_partition[idx].unsqueeze(0)
        if self.augment:
            aug = tio.Compose([tio.OneOf\
                               ({tio.RandomAffine(scales= (0.9, 1.1, 0.9, 1.1, 1, 1),
                                                  degrees= (5.0, 5.0, 0)): 0.35,
                                 tio.RandomElasticDeformation(num_control_points=9,
                                                  max_displacement= (0.1, 0.1, 0.1),
                                                  locked_borders= 2, 
                                                  image_interpolation= 'linear'): 0.35,
                                 tio.RandomFlip(axes=(2,)):.3}),
                              ])
            subject = tio.Subject(ct=tio.ScalarImage(tensor = image),
                                  mask=tio.ScalarImage(tensor = mask))
            output = aug(subject)
            augmented_image = output['ct']
            augmented_mask = output['mask']
            image = augmented_image.data
            mask = augmented_mask.data
        # note that mask is integer
        mask = mask.type(torch.IntTensor)        
        image = image.type(torch.FloatTensor)

        #The tensor we pass into ScalarImage is C x W x H x D, so permute axes to
        # C x D x H x W. At the end we have N x 1 x D x H x W.
        image = image.permute(0,3,2,1)
        mask = mask.permute(0,3,2,1)
        
        # Return image and mask pair tensors
        return image, mask

