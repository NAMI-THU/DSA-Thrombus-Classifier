#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 13:04:52 2020

@author: nami
"""

# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 16:56:53 2019

@author: mittmann
"""
#import cProfile
#import pstats
#from pstats import SortKey

from DsaDataset import DsaDataset
from torch.utils.data import DataLoader
from ModelEvaluation import ModelEvaluation
from CnnLstmModel import CnnLstmModel
import torchvision.models as models
import torch.cuda
import torch.optim
import torch.nn
from torch import autograd
import matplotlib as plt
import numpy as np
import gc

import torch.cuda.amp as amp

THROMBUS_NO = 0.214
THROMBUS_YES = 0.786

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.pyplot.plot(ave_grads, alpha=0.3, color="b")
    plt.pyplot.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.pyplot.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.pyplot.xlim(xmin=0, xmax=len(ave_grads))
    plt.pyplot.xlabel("Layers")
    plt.pyplot.ylabel("average gradient")
    plt.pyplot.title("Gradient flow")
    plt.pyplot.grid(True)
    plt.pyplot.show()

if __name__ == "__main__":
    
    torch.set_num_threads(8)    
    
    PATH = "/media/nami/TE_GZ/Datenauswertung_DSA-Bilder/CrossFoldValidation/Models/CnnLstm/Regnet_y_16gf_Gru_resize/"
    data_path = "/media/nami/TE_GZ/DSA-aufbereitet-nifti"
    csv_path = "/media/nami/TE_GZ/Datenauswertung_DSA-Bilder/CrossFoldValidation/Datasets/"


    
    for fold in range(1, 6):
        torch.cuda.empty_cache()
        print('Fold {}'.format(fold))
        
        modelEvaluationTrain = ModelEvaluation()
        modelEvaluationTest = ModelEvaluation()
    
        #data_set_train = DsaDataset(data_path, csv_path, "Dataset_4_fold" + str(fold) + "_train.csv", training=True)
        data_set_train = DsaDataset(data_path, csv_path, "Dataset_4_gesamt.csv", training=True)
        data_set_train.loadCsvData()    
        data_set_train.createDatasetDict()
        
    
        data_set_test = DsaDataset(data_path, csv_path, "Dataset_4_fold" + str(fold) + "_valid.csv", training=False)
        data_set_test.loadCsvData()
        data_set_test.createDatasetDict()
    
        batchSize = 1
        dataLoaderTrain = DataLoader(dataset=data_set_train, batch_size=batchSize, shuffle=True, sampler=None,
                                batch_sampler=None, num_workers=4, collate_fn=None,
                                pin_memory=False, drop_last=False, #timeout=2400,
                                worker_init_fn=None)
        
        dataLoaderTest = DataLoader(dataset=data_set_test, batch_size=1, shuffle=False, sampler=None,
                                batch_sampler=None, num_workers=4, collate_fn=None,
                                pin_memory=False, drop_last=False, #timeout=2400,
                                worker_init_fn=None)
    
        device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device2 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    
        #Load Checkpoints:   
        #-if fold == 2:
        #-    checkpoint_frontal = torch.load(PATH + "frontal_last_fold_2.pt")        
        #-    checkpoint_lateral = torch.load(PATH + "lateral_last_fold_2.pt")
        #-    checkpoint_best_mcc_frontal = torch.load(PATH + "frontal_best_mcc_fold_2.pt")        
        #-    checkpoint_best_mcc_lateral = torch.load(PATH + "lateral_best_mcc_fold_2.pt")
        
        starting_epoch = 0 #-checkpoint_frontal['epoch'] + 1 if fold == 2 else 0
        
        
        #Initialize CNN for frontal images:
        model_frontal = CnnLstmModel(512, 3, 1, True, device1)
        #-if fold == 2:
        #-    model_frontal.load_state_dict(checkpoint_frontal['model_state_dict'])
        model_frontal.to(device1)

        
        optimizer_frontal = torch.optim.AdamW(model_frontal.parameters(), lr=0.00001, weight_decay=0.01) 
        scheduler_frontal = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_frontal, 'min', factor=0.1, patience=20, verbose=True)
        loss_function_frontal = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1))
         
        
        #Initialize CNN for lateral images:
        model_lateral = CnnLstmModel(512, 3, 1, True, device2)
        #-if fold == 2:
        #-    model_lateral.load_state_dict(checkpoint_lateral['model_state_dict'])
        model_lateral.to(device2)
    
        
        optimizer_lateral = torch.optim.AdamW(model_lateral.parameters(), lr=0.00001, weight_decay=0.01)
        scheduler_lateral = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_lateral, 'min', factor=0.1, patience=20, verbose=True)
        loss_function_lateral = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1))
                
        scaler_frontal = amp.GradScaler()
        scaler_lateral = amp.GradScaler()
        
        
        best_loss_frontal = 0.0
        best_mcc_frontal = -1.0 #-if fold != 2 else checkpoint_best_mcc_frontal['mcc']
        running_loss_frontal = 0.0
        
        best_loss_lateral = 0.0
        best_mcc_lateral = -1.0 #-if fold != 2 else checkpoint_best_mcc_lateral['mcc']
        running_loss_lateral = 0.0
        
        
        loss_function_validation = torch.nn.BCEWithLogitsLoss()
        
        #-if fold == 2:
        #-    del checkpoint_frontal, checkpoint_lateral, checkpoint_best_mcc_frontal, checkpoint_best_mcc_lateral
        
        
        for epoch in range(starting_epoch, 130):
            model_frontal.train()
            model_lateral.train()
            model_frontal.requires_grad_(True)
            model_lateral.requires_grad_(True)
            #model_frontal.cnn.requires_grad_(False)
            #model_lateral.cnn.requires_grad_(False)
            modelEvaluationTrain.reset()
            
            
            #for name, module in model_frontal.named_modules():
            #    print(name)
            #    print(module)
            '''    
            count = 0
            for child in model_frontal.cnn.children():
                print(child)
                count += 1
                if count < 2:
                    for param in child.parameters():
                        param.requires_grad = False
            count = 0
            for child in model_lateral.cnn.children():
                count += 1
                if count < 2:
                    for param in child.parameters():
                        param.requires_grad = False
            '''
            
            count_equal = 0
            count_not_equal = 0
            
            for step, batch in enumerate(dataLoaderTrain):
                
                #if step >= 20:
                #    break
                
                hasThrombus_frontal = torch.zeros((batch['keypoints'].shape[0],1))
                for index1, keypoint1 in enumerate(batch['keypoints']):
                    hasThrombus_frontal[index1] = THROMBUS_NO if torch.max(keypoint1) == 0 else THROMBUS_YES
                    
                hasThrombus_lateral = torch.zeros((batch['keypointsOtherView'].shape[0],1))
                for index2, keypoint2 in enumerate(batch['keypointsOtherView']):
                    hasThrombus_lateral[index2] = THROMBUS_NO if torch.max(keypoint2) == 0 else THROMBUS_YES
                    # WICHTIG:
                    # Beim Extraktionsprozess noch bedenken, dass eine Ansicht als Thrombusfrei
                    # markiert ist, die andere Ansicht aber als nicht thrombusfrei 
                    # Das wird im folgenden überprüft und entsprechend behandelt:
                
                for index in range(len(hasThrombus_frontal)):
                    if not torch.equal(hasThrombus_frontal[index], hasThrombus_lateral[index]):
                        hasThrombus_frontal[index] = THROMBUS_YES
                        hasThrombus_lateral[index] = THROMBUS_YES     
                        count_not_equal += 1
                    else:
                        count_equal += 1
                

                labels_frontal = hasThrombus_frontal.to(device=device1, dtype=torch.float)
                images_frontal = batch['image'].to(device=device1, dtype=torch.float)
                
                labels_lateral = hasThrombus_lateral.to(device=device2, dtype=torch.float)
                images_lateral = batch['imageOtherView'].to(device=device2, dtype=torch.float)
                
                
                optimizer_frontal.zero_grad()
                optimizer_lateral.zero_grad()
                
                with amp.autocast():
                    output_frontal = model_frontal(images_frontal)
                    output_lateral = model_lateral(images_lateral)
                                        
                    loss_frontal = loss_function_frontal(output_frontal, labels_frontal)
                    loss_lateral = loss_function_lateral(output_lateral, labels_lateral)
                

                del images_frontal
                del images_lateral
                
                #print(torch.cuda.memory_summary())
                
                scaler_frontal.scale(loss_frontal).backward()
                scaler_lateral.scale(loss_lateral).backward()
                

                del labels_frontal
                del labels_lateral
                scaler_frontal.step(optimizer_frontal)
                scaler_lateral.step(optimizer_lateral)
                
                scaler_frontal.update()
                scaler_lateral.update()
                
                running_loss_frontal += loss_frontal.detach().item()           
                running_loss_lateral += loss_lateral.detach().item()
            
                
                if epoch == 0 and fold == 1 and (step == 1 or step == 15):
                    plot_grad_flow(model_frontal.named_parameters())
    
                #--scheduler_frontal.step()
                #--scheduler_lateral.step()
                
                #-------------- Evaluate Training ACC PREC and Recall ------------
                estimate_train_frontal = THROMBUS_NO if torch.sigmoid(output_frontal).item() <= 0.5 else THROMBUS_YES
                estimate_train_lateral = THROMBUS_NO if torch.sigmoid(output_lateral).item() <= 0.5 else THROMBUS_YES
                
                if estimate_train_frontal == THROMBUS_NO:
                    if torch.max(batch['keypoints']) == 0:
                        modelEvaluationTrain.increaseTNfrontal()
                    else:
                        modelEvaluationTrain.increaseFNfrontal()
                else: # means: estimate_frontal = 1
                    if torch.max(batch['keypoints']) == 0:
                        modelEvaluationTrain.increaseFPfrontal()
                    else:
                        modelEvaluationTrain.increaseTPfrontal()
                
                if estimate_train_lateral == THROMBUS_NO:
                    if torch.max(batch['keypointsOtherView']) == 0:
                        modelEvaluationTrain.increaseTNlateral()
                    else:
                        modelEvaluationTrain.increaseFNlateral()
                else: # means: estimate_lateral = 1
                    if torch.max(batch['keypointsOtherView']) == 0:
                        modelEvaluationTrain.increaseFPlateral()
                    else:
                        modelEvaluationTrain.increaseTPlateral()
                
                #torch.cuda.empty_cache()

            #------------- Ende for loop training ---------------------------------
            #print(count_equal)
            #print(count_not_equal)
            '''
            # ------------- Schedular Steps ---------------------------------------
            scheduler_frontal.step(running_loss_frontal / (step + 1))
            scheduler_lateral.step(running_loss_lateral / (step + 1))
            #------------- Print Loss statistics and save models ------------------
            print('Epoche {}'.format(epoch))
            print('loss_frontal = {} ; loss_lateral = {}'.format(running_loss_frontal / (step + 1), running_loss_lateral / (step + 1) ))
            modelEvaluationTrain.printAllStats()
            
            running_loss_frontal = 0.0
            running_loss_lateral = 0.0
            
            path_frontal = PATH + 'frontal_last_fold_' + str(fold) + '.pt'
            torch.save({
            'epoch': epoch,
            'model_state_dict': model_frontal.state_dict(),
            'optimizer_state_dict': optimizer_frontal.state_dict(),
            'loss': loss_frontal}, path_frontal)
            
            path_lateral = PATH + 'lateral_last_fold_' + str(fold) + '.pt'
            torch.save({
            'epoch': epoch,
            'model_state_dict': model_lateral.state_dict(),
            'optimizer_state_dict': optimizer_lateral.state_dict(),
            'loss': loss_lateral}, path_lateral)


            #----------------- Evaluate Model -------------------------------------
            model_frontal.eval()
            model_lateral.eval()
            model_frontal.requires_grad_(False)
            model_lateral.requires_grad_(False)  
            
            modelEvaluationTest.reset()
            validation_loss_frontal = 0
            validation_loss_lateral = 0
            
            for step, batch in enumerate(dataLoaderTest):
                
                #if step >= 3:
                #    break
            
                hasThrombus_frontal = torch.tensor([[THROMBUS_NO]]) if torch.max(batch['keypoints']) == 0 else torch.tensor([[THROMBUS_YES]])      
                hasThrombus_lateral = torch.tensor([[THROMBUS_NO]]) if torch.max(batch['keypointsOtherView']) == 0 else torch.tensor([[THROMBUS_YES]])
                
                for index in range(len(hasThrombus_frontal)):
                    if not torch.equal(hasThrombus_frontal[index], hasThrombus_lateral[index]):
                        hasThrombus_frontal[index] = THROMBUS_YES
                        hasThrombus_lateral[index] = THROMBUS_YES
                        
                labels_frontal = hasThrombus_frontal.to(device=device1, dtype=torch.float)
                labels_lateral = hasThrombus_lateral.to(device=device2, dtype=torch.float)
                
                images_frontal = batch['image']#.to(device=device, dtype=torch.float)
                images_lateral = batch['imageOtherView']#.to(device=device, dtype=torch.float)
    
                output_frontal = model_frontal(images_frontal)
                output_lateral = model_lateral(images_lateral)
    
                
                validation_loss_frontal += loss_function_validation(output_frontal, labels_frontal).item()
                validation_loss_lateral += loss_function_validation(output_lateral, labels_lateral).item()
                
                
                del images_frontal
                del images_lateral
                
                
                #------------- Evaluate Validation ACC PREC and Recall ------------
                estimate_frontal = THROMBUS_NO if torch.sigmoid(output_frontal).item() <= 0.5 else THROMBUS_YES
                estimate_lateral = THROMBUS_NO if torch.sigmoid(output_lateral).item() <= 0.5 else THROMBUS_YES
                
                if estimate_frontal == THROMBUS_NO:
                    if torch.max(batch['keypoints']) == 0:
                        modelEvaluationTest.increaseTNfrontal()
                    else:
                        modelEvaluationTest.increaseFNfrontal()
                else: # means: estimate_frontal = 1
                    if torch.max(batch['keypoints']) == 0:
                        modelEvaluationTest.increaseFPfrontal()
                    else:
                        modelEvaluationTest.increaseTPfrontal()
                
                if estimate_lateral == THROMBUS_NO:
                    if torch.max(batch['keypointsOtherView']) == 0:
                        modelEvaluationTest.increaseTNlateral()
                    else:
                        modelEvaluationTest.increaseFNlateral()
                else: # means: estimate_lateral = 1
                    if torch.max(batch['keypointsOtherView']) == 0:
                        modelEvaluationTest.increaseFPlateral()
                    else:
                        modelEvaluationTest.increaseTPlateral()
                        
                      
            # ------------- Ende for loop validation data loader ------------------
            print('val_loss_frontal = {} ; val_loss_lateral = {}'.format(validation_loss_frontal / (step + 1), validation_loss_lateral / (step + 1)))
            modelEvaluationTest.printAllStats()
    
            
            if best_mcc_frontal == -1.0:
                best_mcc_frontal = modelEvaluationTest.getMccFrontal()
                path_frontal = PATH + 'frontal_best_mcc_fold_' + str(fold) + '.pt'
                torch.save({
                'epoch': epoch,
                'model_state_dict': model_frontal.state_dict(),
                'optimizer_state_dict': optimizer_frontal.state_dict(),
                'loss': validation_loss_frontal / (step + 1),
                'mcc': best_mcc_frontal,
                'acc': modelEvaluationTest.getAccuracyFrontal()}, path_frontal)
                
            elif best_mcc_frontal < modelEvaluationTest.getMccFrontal():
                best_mcc_frontal = modelEvaluationTest.getMccFrontal()
                path_frontal = PATH + 'frontal_best_mcc_fold_' + str(fold) + '.pt'
                torch.save({
                'epoch': epoch,
                'model_state_dict': model_frontal.state_dict(),
                'optimizer_state_dict': optimizer_frontal.state_dict(),
                'loss': validation_loss_frontal / (step + 1),
                'mcc': best_mcc_frontal,
                'acc': modelEvaluationTest.getAccuracyFrontal()}, path_frontal)
                
            
            if best_mcc_lateral == -1.0:
                best_mcc_lateral = modelEvaluationTest.getMccLateral()
                path_lateral = PATH + 'lateral_best_mcc_fold_' + str(fold) + '.pt'
                torch.save({
                'epoch': epoch,
                'model_state_dict': model_lateral.state_dict(),
                'optimizer_state_dict': optimizer_lateral.state_dict(),
                'loss': validation_loss_lateral / (step + 1),
                'mcc': best_mcc_lateral,
                'acc': modelEvaluationTest.getAccuracyLateral()}, path_lateral)
                
            elif best_mcc_lateral < modelEvaluationTest.getMccLateral():
                best_mcc_lateral = modelEvaluationTest.getMccLateral()
                path_lateral = PATH + 'lateral_best_mcc_fold_' + str(fold) + '.pt'
                torch.save({
                'epoch': epoch,
                'model_state_dict': model_lateral.state_dict(),
                'optimizer_state_dict': optimizer_lateral.state_dict(),
                'loss': validation_loss_lateral / (step + 1),
                'mcc': best_mcc_lateral,
                'acc': modelEvaluationTest.getAccuracyLateral()}, path_lateral)
        

            '''