import torch
import torch.utils.data
from torch.utils.data import Dataset
import albumentations as A
import torchio as tio

class Pancreas_2D_dataset(Dataset):
    #constructs dataset for 2D model
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
          Because the input image is scaled to [0.0, 1.0], totensor transformation should not be used
          when transforming target image masks.
          https://github.com/pytorch/vision/blob/master/references/segmentation/transforms.py
          """
          mask = torch.as_tensor(augmented['mask']) 
      else:
          image = t1(image)
          mask = torch.as_tensor(mask)

      mask = torch.unsqueeze(mask,0)
      image = image.type(torch.FloatTensor)
      mask = mask.type(torch.IntTensor)
      return image, mask

class Pancreas_3D_dataset(Dataset):    
    #construct dataset for 3D model
    def __init__(self, CT_partition, mask_partition, augment = False):
        self.CT_partition = CT_partition
        self.mask_partition = mask_partition
        self.augment = augment
       
    def __len__(self):
        return (len(self.CT_partition))
    
    def __getitem__(self, idx):      
        #ScalarImage expect 4DTensor
        image = self.CT_partition[idx].unsqueeze(0)
        mask = self.mask_partition[idx].unsqueeze(0)
        if self.augment:
            aug = tio.Compose([tio.OneOf({tio.RandomAffine(scales= (0.9, 1.1, 0.9, 1.1, 1, 1),
                                                      degrees= (5.0, 5.0, 0)): 0.35,
                                          tio.RandomElasticDeformation(num_control_points=9,
                                                      max_displacement= (0.1, 0.1, 0.1),
                                                      locked_borders= 2, 
                                                      image_interpolation= 'linear'): 0.35,
                                          tio.RandomFlip(axes=(2,)):.3}),
                                      ])
            subject = tio.Subject(ct=tio.ScalarImage(tensor = image), mask=tio.ScalarImage(tensor = mask))
            output = aug(subject)
            augmented_image = output['ct']
            augmented_mask = output['mask']
            image = augmented_image.data
            mask = augmented_mask.data

        
        mask = mask.type(torch.IntTensor)        
        image = image.type(torch.FloatTensor)
        #The tensor we pass into ScalarImage is C x W x H x D, so permute axes to C x D x H x W. At the end we have N x 1 x D x H x W
        image = image.permute(0,3,2,1)
        mask = mask.permute(0,3,2,1)
        return image, mask   

