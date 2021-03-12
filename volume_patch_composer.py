#-----------------------------------------------------------------------#
#                          Library imports                              #
#-----------------------------------------------------------------------#
from PIL import Image
import torch
import torch.nn.functional as F
import os
import numpy as np

#-----------------------------------------------------------------------#
#                          volume_composer                              #
#     Create and save resized 3D images and masks from their            #
#                respective stack of 2D slices                          #
#-----------------------------------------------------------------------#
# Adopted from:                                                         #
# https://pytorch.org/docs/stable/torchvision/transforms.html           #
# https://github.com/pytorch/vision/blob/master/references/\            #
#       segmentation/transforms.py                                      #
#-----------------------------------------------------------------------#
# p:                    patient number                                  #                                        
# patient_image_cnt_CT: a list of the number of slices for each patient #                                        
# patient_path_list:    a list of the CT and mask slices paths for each #
#                       patient.                                        #      
# grid:                 flow-field, the gridsample computes the output  #
#                       using input pixel locations from the grid,      #
#                       (height, width,depth)                           #
# CT_3D_list:           a list of 3D presentation of each CT slice      #
# masks_3D_list:        a list of 3D presentation of each mask slice    #
# CT_3D, mask_3D:       resized 3D CT and mask in torch tensor          #
#-----------------------------------------------------------------------#
def volume_composer(p, patient_image_cnt_CT, patient_path_list, grid):
    #resize the CT and mask image
    CT_3D_list = []
    masks_3D_list = []
    for s in range(patient_image_cnt_CT[p]):
        image = Image.open(patient_path_list ['CT'][p][s]) 
        mask = Image.open(patient_path_list ['Masks'][p][s]) 
        # create numpy array objects from the CT and mask Image objects
        image = np.array(image)
        mask = np.array(mask)
        # add a new dimension to the mask and CT slice
        image = image[np.newaxis, :, :] 
        mask = mask[np.newaxis, :, :] 
        # add the 3D CTs and masks to their lists, note that they are 3D presentation of each slice
        CT_3D_list.append(image)
        masks_3D_list.append(mask)
    # stack 3D slices, build 3D numpy objects and convert them into torch tensors
    image = torch.as_tensor(np.vstack(CT_3D_list))
    image = image.unsqueeze(0).unsqueeze(0)
    """
    ref: https://pytorch.org/docs/stable/torchvision/transforms.html
    Because the input image is scaled to [0.0, 1.0], totensor transformation should not be used
    when transforming target image masks.
    ref: https://github.com/pytorch/vision/blob/master/references/segmentation/transforms.py
    """
    mask = torch.as_tensor(np.vstack(masks_3D_list))  
    mask = mask.unsqueeze(0).unsqueeze(0)
    image = image.type(torch.FloatTensor)
    mask = mask.type(torch.FloatTensor)
    # create resized 3D CT and mask tensors
    # for the mask, to have either 0 or 1 as the voxel value, use 'nearest' for the interpoaltion mode
    CT_3D = F.grid_sample(image, grid, mode = 'bilinear', align_corners=True)
    mask_3D = F.grid_sample(mask, grid, mode ='nearest', align_corners=True)
    mask_3D = mask_3D.type(torch.IntTensor)
    # save the patient's resized 3D CT and mask
    torch.save(CT_3D, '/content/data3D/' + p + '_CT.pt' )
    torch.save(mask_3D, '/content/data3D/' + p + '_Mask.pt' )

#-----------------------------------------------------------------------#
#                          patch_creator                                #
#     Create and save resized 3D images and masks from their            #
#                respective stack of 2D slices                          #
#-----------------------------------------------------------------------#
# Adopted from:                                                         #
# https://discuss.pytorch.org/t/\                                       #
#             patch-making-does-pytorch-have-anything-to-offer/33850/11 #
# https://discuss.pytorch.org/t/fold-unfold-to-get-patches/53419        #
#-----------------------------------------------------------------------#
# Returns lists of torch CT and mask subvolumes for each partition      #
#-----------------------------------------------------------------------#
# partition:    either train, valid, or test partition                  #
# kc, kh, kw:   kernel size (patch parameters for volumetric            #
#               segmentation)                                           #
# dc, dh, dw:   stride (patch parameters for volumetric segmentation)   #
# p:            patient number                                          #                                        
# CT_patches:   a list of CT subvolumes in torch tensor                 #
# mask_patches: a list of mask subvolumes in torch tensor               #
#-----------------------------------------------------------------------#
def patch_creator(partition, kw, kh, kc, dw, dh, dc):
    # create 3D CT and mask patches (subvolumes)
    CT_patches =[]
    mask_patches = []
    for p in partition:
        CT_pth = os.path.join('/content', 'data3D', p + '_CT.pt')  
        mask_pth = os.path.join('/content', 'data3D', p + '_Mask.pt')
        ct = torch.load(CT_pth)
        mask = torch.load(mask_pth)
        ct = ct.squeeze(0).squeeze(0)
        mask = mask.squeeze(0).squeeze(0)
        # create subvolumes
        #it is like folding along width, then heigth, then depth
        #for a [1, 1, 256, 256, 128] tensor:squeezing->[256,256,128]
        CT_patch = ct.unfold(0,kw, dw)  # -->[4, 256, 128, 64]
        CT_patch = CT_patch.unfold(1,kh, dh) #-->[4, 4, 128, 64, 64]
        CT_patch = CT_patch.unfold(2,kc, dc) #-->[4, 4, 4, 64, 64, 32]
        mask_patch = mask.unfold(0,kw, dw)
        mask_patch = mask_patch.unfold(1,kh, dh)
        mask_patch = mask_patch.unfold(2,kc, dc)  
        # add each patient's CT and mask subvolumes to their corresponding list
        CT_patches.extend(CT_patch.contiguous().view(-1, kw, kh, kc))
        mask_patches.extend(mask_patch.contiguous().view(-1, kw, kh, kc))
    return CT_patches, mask_patches

