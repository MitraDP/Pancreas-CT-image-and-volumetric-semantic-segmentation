from PIL import Image
import torch
import torch.nn.functional as F
import os
import numpy as np

def volume_composer(p, patient_image_cnt_CT, patient_path_list, grid):
    #resize the CT and mask image
    CT_3D_list = []
    masks_3D_list = []
    for s in range(patient_image_cnt_CT[p]):
      image = Image.open(patient_path_list ['CT'][p][s]) 
      mask = Image.open(patient_path_list ['Masks'][p][s]) 
      image = np.array(image)
      mask = np.array(mask)
      #add a new dimension to the mask and CT slice
      image = image[np.newaxis, :, :] 
      mask = mask[np.newaxis, :, :] 
      CT_3D_list.append(image)
      masks_3D_list.append(mask)
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
    #for the mask, to have either 0 or 1 as the voxel value, use 'nearest' for the interpoaltion mode
    CT_3D = F.grid_sample(image, grid, mode = 'bilinear', align_corners=True)
    mask_3D = F.grid_sample(mask, grid, mode ='nearest', align_corners=True)
    mask_3D = mask_3D.type(torch.IntTensor)
    torch.save(CT_3D, '/content/data3D/' + p + '_CT.pt' )
    torch.save(mask_3D, '/content/data3D/' + p + '_Mask.pt' )

    
def patch_creator(partition, kw, kh, kc, dw, dh, dc):
    #create 3D CT and mask patches (subvolumes)
    CT_patches =[]
    mask_patches = []
    for p in partition:
        CT_pth = os.path.join('/content', 'data3D', p + '_CT.pt')  
        mask_pth = os.path.join('/content', 'data3D', p + '_Mask.pt')
        ct = torch.load(CT_pth)
        mask = torch.load(mask_pth)
        ct = ct.squeeze(0).squeeze(0)
        mask = mask.squeeze(0).squeeze(0)
        CT_patch = ct.unfold(0,kw, dw)
        CT_patch = CT_patch.unfold(1,kh, dh)
        CT_patch = CT_patch.unfold(2,kc, dc)

        mask_patch = mask.unfold(0,kw, dw)
        mask_patch = mask_patch.unfold(1,kh, dh)
        mask_patch = mask_patch.unfold(2,kc, dc)  
        CT_patches.extend(CT_patch.contiguous().view(-1, kw, kh, kc))
        mask_patches.extend(mask_patch.contiguous().view(-1, kw, kh, kc))
    return CT_patches, mask_patches

