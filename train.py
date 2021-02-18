import torch
from tqdm import tqdm
import numpy as np
import pandas as pd

def train_2D(n_epochs, loaders, model, optimizer, criterion, train_on_gpu, performance_metrics, path):
    #train 2D UNet for some number of epochs
    #keep track of loss and performance merics
    loss_and_metrics =[]
    # initialize tracker for max DSC 
    DSC_max = 0
    show_every = 500
    # epoch training loop
    for epoch in tqdm( range(1, n_epochs+1), total = n_epochs+1):
        print(f'=== Epoch #{epoch} ===')
        # initialize variables to monitor training and validation loss, and performance metrics
        train_loss = 0.0
        valid_loss = 0.0

        specificity_val = 0
        sensitivity_val = 0
        precision_val = 0
        F1_score_val = 0
        F2_score_val = 0
        DSC_val = 0
        valid_cnt = 0
        ###################
        # train the model #
        ###################
        model.train()
        print('=== Training ===')
        # batch training loop
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            if batch_idx % show_every == 0:
                print(f'{batch_idx + 1} / {len(loaders["train"])}...')
            # clear the gradients of all optimized variable
            optimizer.zero_grad() 
            # forward pass (inference)
            output = model(data) 
            # calculate the batch loss
            loss = criterion(output, target) 
            # backpropagation
            loss.backward() 
            # Update weights
            optimizer.step() 
            # update training loss
            train_loss += ((1 / (batch_idx + 1)) * (loss.data - train_loss)) 
            
                         
        ######################    
        # validate the model #
        ######################
        print('=== Validation ===')
        model.eval()
        with torch.no_grad():
            # batch training loop
            for batch_idx, (data, target) in enumerate(loaders['valid']):
                if batch_idx % show_every == 0:
                    print(f'{batch_idx + 1} / {len(loaders["valid"])}...')
                # move to GPU
                if train_on_gpu:
                    data, target = data.cuda(), target.cuda()
                # forward pass (inference)
                output = model(data)
                # calculate the batch loss
                loss = criterion (output, target)
                # update validation loss
                valid_loss +=  ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
                # convert output probabilities to predicted class
                output = output.cpu().detach().numpy()
                output_b = (output>0.5)*1
                output_b = np.squeeze(output_b)
                batch_l = output_b.size
                valid_cnt += batch_l
                t1 = transforms.ToTensor()
                output_b = t1(output_b)
                output_b = output_b.cuda()
                # calculate performance metrics per batches
                m = performance_metrics(smooth = 1e-6)
                specificity, sensitivity, precision, F1_score, F2_score, DSC =  m(target, output_b)    
                
                specificity_val += specificity * batch_l
                sensitivity_val += sensitivity * batch_l
                precision_val += precision * batch_l
                F1_score_val += F1_score * batch_l
                F2_score_val += F2_score * batch_l 
                DSC_val += DSC * batch_l 
            
        specificity_val, sensitivity_val, precision_val, F1_score_val, F2_score_val, DSC_val = specificity_val/valid_cnt, sensitivity_val/valid_cnt, precision_val/valid_cnt, F1_score_val/valid_cnt, F2_score_val/valid_cnt, DSC_val/valid_cnt

        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.4f} \tValidation Loss: {:.4f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        print('Specificity: {:.6f} \tSensitivity: {:.6f} \tF2_score: {:.6f} \tDSC: {:.6f}'.format(
            specificity_val,
            sensitivity_val, 
            F2_score_val, 
            DSC_val
        ))
        
        
        if DSC_val > DSC_max:
            print('Validation DSC increased.  Saving model ...')            
            torch.save(model.state_dict(), path)
            torch.save(model.state_dict(), '/content/drive/MyDrive/checkpoint.pt')
            DSC_max = DSC_val

        loss_and_metrics.append((epoch, train_loss.cpu().detach().numpy(), valid_loss.cpu().detach().numpy(), specificity_val.cpu().detach().numpy(), sensitivity_val.cpu().detach().numpy(), precision_val.cpu().detach().numpy(), F1_score_val.cpu().detach().numpy(), F2_score_val.cpu().detach().numpy(), DSC_val.cpu().detach().numpy() ))

    #save the loss_epoch history
    df=pd.DataFrame.from_records(loss_and_metrics, columns=['epoch', 'Training Loss', 'Validation Loss', 'specificity', 'sensitivity', 'precision', 'F1_score', 'F2_score', 'DSC' ])
    df.to_csv('performance_metrics.csv', index=False)      
    get_ipython().system('cp performance_metrics.csv   /content/drive/MyDrive/performance_metrics.csv')
    torch.save(model.state_dict(), '/content/drive/MyDrive/checkpoint.pt') 
    # return trained model
    return model

def train_3D(n_epochs, loaders, model, optimizer, criterion, train_on_gpu, performance_metrics, path):
    #train 3D UNet for some number of epochs
    #keep track of loss and performance merics
    loss_and_metrics =[]
    # initialize tracker for max DSC 
    DSC_max = 0
    show_every = 50
    for epoch in tqdm( range(1, n_epochs+1), total = n_epochs+1):
        print(f'=== Epoch #{epoch} ===')
        # initialize variables to monitor training and validation loss, and performance metrics
        train_loss = 0.0
        valid_loss = 0.0

        specificity_val = 0
        sensitivity_val = 0
        precision_val = 0
        F1_score_val = 0
        F2_score_val = 0
        DSC_val = 0
        valid_cnt = 0
        ###################
        # train the model #
        ###################
        model.train()
        print('=== Training ===')
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            if batch_idx % show_every == 0:
                print(f'{batch_idx + 1} / {len(loaders["train"])}...')
            # clear the gradients of all optimized variable
            optimizer.zero_grad() 
            # forward pass (inference)
            output = model(data) 
            # calculate the batch loss
            loss = criterion(output, target) 
            # backpropagation
            loss.backward() 
            # Update weights
            optimizer.step() 
            # update training loss
            train_loss += ((1 / (batch_idx + 1)) * (loss.data - train_loss)) 
            
                         
        ######################    
        # validate the model #
        ######################
        print('=== Validation ===')
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(loaders['valid']):
                if batch_idx % show_every == 0:
                    print(f'{batch_idx + 1} / {len(loaders["valid"])}...')
                # move to GPU
                if train_on_gpu:
                    data, target = data.cuda(), target.cuda()
                # forward pass (inference)
                output = model(data)
                # calculate the batch loss
                loss = criterion (output, target)
                # update validation loss
                valid_loss +=  ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
                # convert output probabilities to predicted class
                output = output.cpu().detach().numpy()
                output_b = (output>0.5)*1
                output_b = np.squeeze(output_b)
                batch_l = output_b.size
                valid_cnt += batch_l
                output_b = torch.as_tensor(output_b)
                output_b = output_b.cuda()
                # calculate performance metrics per batches
                m = performance_metrics(smooth = 1e-6)
                specificity, sensitivity, precision, F1_score, F2_score, DSC =  m(target, output_b)    
                specificity_val += specificity * batch_l
                sensitivity_val += sensitivity * batch_l
                precision_val += precision * batch_l
                F1_score_val += F1_score * batch_l
                F2_score_val += F2_score * batch_l
                DSC_val += DSC * batch_l 
            
        specificity_val, sensitivity_val, precision_val, F1_score_val, F2_score_val, DSC_val = specificity_val/valid_cnt, sensitivity_val/valid_cnt, precision_val/valid_cnt, F1_score_val/valid_cnt, F2_score_val/valid_cnt, DSC_val/valid_cnt

        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.4f} \tValidation Loss: {:.4f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        print('Specificity: {:.6f} \tSensitivity: {:.6f} \tF2_score: {:.6f} \tDSC: {:.6f}'.format(
            specificity_val,
            sensitivity_val, 
            F2_score_val, 
            DSC_val
        ))
        
        
        if DSC_val > DSC_max:
            print('Validation DSC increased.  Saving model ...')            
            torch.save(model.state_dict(), path)
            torch.save(model.state_dict(), '/content/drive/MyDrive/checkpoint.pt')
            DSC_max = DSC_val

        loss_and_metrics.append((epoch, train_loss.cpu().detach().numpy(), valid_loss.cpu().detach().numpy(), specificity_val.cpu().detach().numpy(), sensitivity_val.cpu().detach().numpy(), precision_val.cpu().detach().numpy(), F1_score_val.cpu().detach().numpy(), F2_score_val.cpu().detach().numpy(), DSC_val.cpu().detach().numpy() ))

    #save the loss_epoch history
    df=pd.DataFrame.from_records(loss_and_metrics, columns=['epoch', 'Training Loss', 'Validation Loss', 'specificity', 'sensitivity', 'precision', 'F1_score', 'F2_score', 'DSC' ])
    df.to_csv('performance_metrics.csv', index=False)      
    get_ipython().system('cp performance_metrics.csv   /content/drive/MyDrive/performance_metrics.csv')
    torch.save(model.state_dict(), '/content/drive/MyDrive/checkpoint.pt') 
    # return trained model
    return model

