#-----------------------------------------------------------------------#
#                          Library imports                              #
#-----------------------------------------------------------------------#
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from volume_patch_composer import  patch_creator
from metrics import performance_metrics
import nibabel as nib



#-----------------------------------------------------------------------#
#             get_inference_performance_metrics_2D                      #
#  Performs prediction on the test dataset, return the performance      #
#  metrics for each patient                                             #
#-----------------------------------------------------------------------#
# Returns inference metrics table                                       #
#-----------------------------------------------------------------------#
# model:         Trained model                                          #  
# part:          A list of patients in the test partition               #
# dataset_test:  Test dataset which is grouped per patient              #
# threshold:     Threshold value to create binary image                 #
#-----------------------------------------------------------------------#
def get_inference_performance_metrics_2D(model, part, dataset_test,
                                         batch_size, train_on_gpu, threshold):
    # Initialize a list to keep track of test performance metrics    
    test_metrics =[]
    
    # Set the model to inference mode
    model.eval()
    for p in part:
        # Test dataloader per patient
        loaders = torch.utils.data.DataLoader(dataset_test[p], 
                                                      batch_size=batch_size, 
                                                      shuffle=False, 
                                                      num_workers= 0)
        
        # Initialize variables to monitor performance metrics
        specificity_val = 0
        sensitivity_val = 0
        precision_val = 0
        F1_score_val = 0
        F2_score_val = 0
        DSC_val = 0
        # initialize the number of test instances
        test_cnt = 0
        
        for batch_idx, (data, target) in enumerate(loaders):
            # Move image & mask Pytorch Tensor to GPU if CUDA is available.
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            # forward pass (inference) to get the output
            output = model(data)
            output = output.cpu().detach().numpy()
            # Binarize the output
            output_b = (output>threshold)*1
            output_b = np.squeeze(output_b)
            batch_l = output_b.size
            # Update the total number of inference pairs 
            test_cnt += batch_l
            t1 = transforms.ToTensor()
            # Transform output back to Pytorch Tensor and move it to GPU
            output_b = t1(output_b)
            output_b = output_b.cuda()
            m = performance_metrics(smooth = 1e-6)
            # Get average metrics per batch
            specificity, sensitivity, precision, F1_score, F2_score, DSC = m(
                output_b, target)    
            specificity_val += specificity * batch_l
            sensitivity_val += sensitivity * batch_l
            precision_val += precision * batch_l
            F1_score_val += F1_score * batch_l
            F2_score_val += F2_score * batch_l 
            DSC_val += DSC * batch_l 
            
       
    # Calculate the overall average metrics   
    specificity_val, sensitivity_val, precision_val, F1_score_val, 
    F2_score_val, DSC_val = specificity_val/test_cnt, sensitivity_val/test_cnt,
    precision_val/test_cnt, F1_score_val/test_cnt, F2_score_val/test_cnt, 
    DSC_val/test_cnt
    # Add each patient's prediction metrics to the list
    test_metrics.append((p, specificity_val, sensitivity_val, precision_val,
                         F1_score_val, F2_score_val, DSC_val ))
    #save the test metrics as a table
    df=pd.DataFrame.from_records(test_metrics, columns=[
        'Patient','specificity', 'sensitivity', 'precision', 'F1_score',
        'F2_score', 'DSC'])
    df.to_csv('test_metrics.csv', index=False)       
    #return the inference metrics table
    return df


#-----------------------------------------------------------------------#
#             get_inference_performance_metrics_3D                      #
#  Builds a test dataset and dataloader per patient and performs        #
#  prediction on the test dataset, return the permormance metrics for   #
#  each patient.                                                        #
#-----------------------------------------------------------------------#
# Returns inference metrics table                                       #
#-----------------------------------------------------------------------#
# model:                Trained model                                   #  
# part:                 A list of patients in the test partition        #
# Pancreas_3D_dataset:  3D dataset which is grouped per patient         #
# threshold:            Threshold value to create binary image          #
#-----------------------------------------------------------------------#
def get_inference_performance_metrics_3D(model, part, Pancreas_3D_dataset, 
                                    batch_size, train_on_gpu, threshold,
                                    kw, kh, kc, dw, dh, dc):
    test_metrics = []
    for patient in part:
        # Set the model to inference mode
        model.eval()
        # Create subvolumes (patches) for patient's CT and mask
        CT_patches = []
        mask_patches =[]
        CT_patches, mask_patches = patch_creator([patient], kw, kh, kc, 
                                                  dw, dh, dc) 
        dataset_test= Pancreas_3D_dataset (CT_patches, mask_patches,
                                            augment= False)
        loaders_test = torch.utils.data.DataLoader(dataset_test, 
                                                    batch_size=batch_size, 
                                                    shuffle=False, 
                                                    num_workers=0)
        specificity_val = 0
        sensitivity_val = 0
        precision_val = 0
        F1_score_val = 0
        F2_score_val = 0
        DSC_val = 0
        # initialize the number of test instances
        valid_cnt = 0
        for batch_idx, (data, target) in enumerate(loaders_test):
            # move to GPU
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            # forward pass
            output = model(data)
            output = output.cpu().detach().numpy()
            # Binarize the output
            output_b = (output>threshold)*1
            output_b = np.squeeze(output_b)
            batch_l = output_b.size
            # update the total number of validation pairs
            valid_cnt += batch_l
            #t1 = transforms.ToTensor()
            # Transform output back to Pytorch Tensor and move it to GPU
            #output_b = t1(output_b)
            output_b = torch.as_tensor(output_b)
            output_b = output_b.cuda()
            # calculate average performance metrics per batches
            m = performance_metrics(smooth = 1e-6)
            specificity, sensitivity, precision, F1_score, F2_score, DSC =  m(output_b, target)    
            
            specificity_val += specificity * batch_l
            sensitivity_val += sensitivity * batch_l
            precision_val += precision * batch_l
            F1_score_val += F1_score * batch_l
            F2_score_val += F2_score * batch_l 
            DSC_val += DSC * batch_l 
            # Calculate the overall average metrics    
        specificity_val, sensitivity_val, precision_val, F1_score_val, F2_score_val, DSC_val = specificity_val/valid_cnt, sensitivity_val/valid_cnt, precision_val/valid_cnt, F1_score_val/valid_cnt, F2_score_val/valid_cnt, DSC_val/valid_cnt

        # Add each patient's prediction metrics to the list
        test_metrics.append((patient,specificity_val, sensitivity_val, 
                             precision_val, F1_score_val, 
                             F2_score_val, DSC_val ))
        #save the test metrics as a table
    df=pd.DataFrame.from_records(test_metrics, 
                                 columns=['Patient', 'specificity', 
                                          'sensitivity', 'precision', 
                                          'F1_score', 'F2_score', 'DSC' ])
    df.to_csv('test_metrics.csv', index=False)       
    #return the inference metrics table
    return df

#-----------------------------------------------------------------------#
#                    visualize_patient_prediction_2D                    #
#  Performs prediction on a specific patient of the test dataset,       # 
#  Plot the image trio: image, mask and prediction                      #
#-----------------------------------------------------------------------#
# model:         Trained model                                          #  
# patient:       patient ID/label                                       #
# dataset_test:  Test dataset which is grouped per patient              #
# threshold:     Threshold value to create binary image                 #
#-----------------------------------------------------------------------#
def visualize_patient_prediction_2D(model, patient, dataset_test, batch_size, 
                                 train_on_gpu, threshold):
    loaders_test = torch.utils.data.DataLoader(dataset_test[patient], 
                                                  batch_size=batch_size, 
                                                  shuffle=False,
                                                  num_workers=0)
    # Set the model to inference mode
    model.eval()

    for batch_idx, (data, target) in enumerate(loaders_test):
        # move to GPU
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        output = output.cpu().detach().numpy()
        # Binarize the output
        output_b = (output>threshold)*1
        output_b = np.squeeze(output_b)
        # Plot the image trio: image, mask and prediction
        for gt, pred in zip(target.cpu().numpy(), output_b):
            gt = np.squeeze(gt)
            pred = np.squeeze(pred)
            plt.figure(figsize=(3,6))
            plt.subplot(1,2,1)
            plt.imshow(gt, cmap="gray", interpolation= None)
            plt.subplot(1,2,2)
            plt.imshow(pred, cmap="gray", interpolation= None)
        

#-----------------------------------------------------------------------#
#                              volume                                   #
#  Creates a volume with the the same depth as patch depth. Its height  #
#  and width is the same as the height and width of the resized volumes.#
#-----------------------------------------------------------------------#
# Returns the rth subvolume of image, mask and prediction               #
#-----------------------------------------------------------------------#
# num_patch_width:     number of patches in the width direction         # 
# num_patch_height:    number of patches in the height direction        #
# num_patch_depth:     number of patches in the depth direction         #
# num_batches:         total number of batches                          #
# r:                   the number of the subvolume to be built          #
# CT_subvol:           dictionaries CT patches per batch                #
# mask_subvol:         dictionaries mask patches per batch              #
# predict_subvol:      dictionaries prediction patches per batch        #
# image_vol:           rth subvolume of CT image                        #
# mask_vol:            rth subvolume of mask                            #
# prediction_vol:      rth subvolume of prediction                      #
# kc:                  kernel size in depth direction
#-----------------------------------------------------------------------#
def volume(num_patch_width, num_patch_height, num_patch_depth, num_batches,
           r, CT_subvol, mask_subvol, predict_subvol, kc):
    image_vol = []
    mask_vol =[]
    prediction_vol = [] 
    #sweep in the depth direction
    for k in range(kc):    
        idx= 0
        image = {}
        mask = {}
        prediction = {}
        # sweep in the width and height direction to create layer k of the rth 
        # subvolume horizontally stack the layer k of patches of each bach and
        # then sweep in the height direction and create an array for the kth
        # layer of the final 3D image. Then vertically stack all layers
        # to build a subvolume. Vertically stacking the subvolumes results
        # in a 3D image.
        for q in range(num_batches):
            for j, (im, m, pred)  in enumerate(zip(CT_subvol[q], mask_subvol[q],
                                                   predict_subvol[q])):
                if j%num_patch_depth == r:
                    im = np.squeeze(im).transpose(0,2,1)
                    m = np.squeeze(m).transpose(0,2,1)
                    pred= pred.transpose(0,2,1)
                    image[idx] = im[k,:,:]
                    mask[idx] = m[k,:,:]
                    prediction[idx] = pred[k,:,:]
                    idx+=1
             
        image_vol.append(np.vstack(tuple([np.hstack(tuple([image[num_patch_width*i + j] 
                                                           for j in range(num_patch_height)])) 
                                          for i in range(num_patch_width)])))
        mask_vol.append(np.vstack(tuple([np.hstack(tuple([mask[num_patch_width*i + j]  
                                                      for j in range(num_patch_height)])) 
                                         for i in range(num_patch_width)])))
        prediction_vol.append(np.vstack(tuple([np.hstack(tuple([prediction[num_patch_width*i + j] 
                                                                for j in range(num_patch_height)])) 
                                               for i in range(num_patch_width)])))
        
    return image_vol, mask_vol, prediction_vol


#-----------------------------------------------------------------------#
#               visualize_patient_prediction_3D                         #
#  Creates 3D images of patient's CT, mask, and prediction, and save    #
#  them as nibabel file. Plot sample of cross sections (slices.)        #
#-----------------------------------------------------------------------#
# model:                Trained model                                   #  
# patient:              patient ID/label                                #
# Pancreas_3D_dataset:  3D dataset which is grouped per patient         #
# threshold:            Threshold value to create binary image          #
# kc, kh, kw:           kernel size (patch parameters for volumetric    #
#                       segmentation)                                   #
# dc, dh, dw:           stride (patch parameters for volumetric         #
#                       segmentation)                                   #
# num_patch_width:      number of patches in the width direction        # 
# num_patch_height:     number of patches in the height direction       #
# num_patch_depth:      number of patches in the depth direction        #
# num_batches:          total number of batches                         #
# r:                    the number of the subvolume to be built         #
# CT_subvol:            dictionaries CT patches per batch               #
# mask_subvol:          dictionaries mask patches per batch             #
# predict_subvol:       dictionaries prediction patches per batch       #
# image_vol:            rth subvolume of CT image                       #
# mask_vol:             rth subvolume of mask                           #
# prediction_vol:       rth subvolume of prediction                     #
# image_volume:         3D CT image                                     #
# mask_volume:          3D annotation mask image                        #
# prediction_volume:    3D prediction image                             #
#-----------------------------------------------------------------------#
def visualize_patient_prediction_3D(model, patient, Pancreas_3D_dataset, 
                                    batch_size, train_on_gpu, threshold,
                                    kw, kh, kc, dw, dh, dc):
    # Set the model to inference mode
    model.eval()
    # Create subvolumes (patches) for patient's CT and mask
    CT_patches = []
    mask_patches =[]
    CT_patches, mask_patches = patch_creator([patient], kw, kh, kc, 
                                              dw, dh, dc) 
    dataset_test= Pancreas_3D_dataset (CT_patches, mask_patches,
                                        augment= False)
    loaders_test = torch.utils.data.DataLoader(dataset_test, 
                                                batch_size=batch_size, 
                                                shuffle=False, 
                                                num_workers=0)
    # Create dictionaries of prediction, CT and mask subvolumes per batch
    predict_subvol= {}
    CT_subvol = {}
    mask_subvol ={}

    for batch_idx, (data, target) in enumerate(loaders_test):
        # move to GPU
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass
        output = model(data)
        output = output.cpu().detach().numpy()
        # Binarize the output
        output_b = (output>threshold)*1
        predict_subvol[batch_idx] = np.squeeze(output_b)
        CT_subvol[batch_idx] = np.squeeze(data.cpu().detach().numpy())
        mask_subvol[batch_idx] = np.squeeze(target.cpu().detach().numpy())

    num_batches = 256*256*128 // (kc*kh*kw*batch_size)
    num_patch_depth = 128//kc
    num_patch_width = 256//kw
    num_patch_height = 256//kh
    image_volume = []
    mask_volume =[]
    prediction_volume =[]
    #sweep along the depth direction, create subvolumes and merge them to build 
    #the final 3D image
    for r in range(num_patch_depth):
        image_vol, mask_vol, prediction_vol = volume(num_patch_width, num_patch_height, 
                                                     num_patch_depth, num_batches, 
                                                     r, CT_subvol, mask_subvol,
                                                     predict_subvol, kc)
        image_volume.extend(image_vol)
        mask_volume.extend(mask_vol)
        prediction_volume.extend(prediction_vol)

    nifti_image_np=np.array(image_volume)
    nifti_image = nib.Nifti1Image(nifti_image_np, np.eye(4))  # Save axis for data (just identity)
    nifti_mask_np=np.array(mask_volume)
    nifti_mask = nib.Nifti1Image(nifti_mask_np, np.eye(4))  # Save axis for data (just identity)
    nifti_prediction_np=np.array(prediction_volume).astype('int32')
    nifti_prediction = nib.Nifti1Image(nifti_prediction_np, np.eye(4))  # Save axis for data (just identity)

    nifti_image.header.get_xyzt_units()
    nifti_image.to_filename('image.nii.gz')  # Save as NiBabel file
    nifti_mask.header.get_xyzt_units()
    nifti_mask.to_filename('mask.nii.gz')  # Save as NiBabel file
    nifti_prediction.header.get_xyzt_units()
    nifti_prediction.to_filename('prediction.nii.gz')  # Save as NiBabel file
    
    #plot sample of image cross sections: CT, mask and predictions
    for k in range(0,128,8):
        plt.figure(figsize=(16,16))
        plt.subplot(1,4,1)
        plt.imshow(nifti_image_np[k,:,:])
        plt.title('CT')
        plt.subplot(1,4,2)
        plt.imshow(nifti_image_np[k,:,:])
        plt.imshow(nifti_mask_np[k,:,:], cmap="jet", alpha = 0.3, interpolation= None)  
        plt.title('CT and mask')
        plt.subplot(1,4,3)
        plt.imshow(nifti_image_np[k,:,:])
        plt.imshow(nifti_prediction_np[k,:,:], cmap="jet", alpha = 0.3, interpolation= None)  
        plt.title('CT and prediction')
        plt.subplot(1,4,4)
        plt.imshow(nifti_prediction_np[k,:,:])
        plt.imshow(nifti_mask_np[k,:,:], cmap="jet", alpha = 0.7, interpolation= None)
        plt.title('mask and prediction')