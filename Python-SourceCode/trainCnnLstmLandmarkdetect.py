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
    
    #pr = cProfile.Profile()
    #pr.enable()

    torch.set_num_threads(8)    
    
    PATH = "/media/nami/TE_GZ/Datenauswertung_DSA-Bilder/CnnLSTM_resnet18_landmarkdetect/"
    data_path = "/media/nami/TE_GZ/DSA-aufbereitet-nifti"
    csv_path = "/media/nami/TE_GZ/Datenauswertung_DSA-Bilder"

    modelEvaluationTrain = ModelEvaluation()
    modelEvaluationTest = ModelEvaluation()

    data_set_train = DsaDataset(data_path, csv_path, "Dataset_3_train.csv", training=True)
    data_set_train.loadCsvData()    
    data_set_train.createDatasetDict()
    

    data_set_test = DsaDataset(data_path, csv_path, "Dataset_3_valid.csv", training=False)
    data_set_test.loadCsvData()
    data_set_test.createDatasetDict()

    batchSize = 1
    dataLoaderTrain = DataLoader(dataset=data_set_train, batch_size=batchSize, shuffle=True, sampler=None,
                            batch_sampler=None, num_workers=2, collate_fn=None,
                            pin_memory=False, drop_last=False, #timeout=2400,
                            worker_init_fn=None)
    
    dataLoaderTest = DataLoader(dataset=data_set_test, batch_size=1, shuffle=False, sampler=None,
                            batch_sampler=None, num_workers=2, collate_fn=None,
                            pin_memory=False, drop_last=False, #timeout=2400,
                            worker_init_fn=None)

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device2 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


    #Load Checkpoints:
    #-checkpoint_frontal = torch.load(PATH + "_frontal_19.pt")        
    #-checkpoint_lateral = torch.load(PATH + "_lateral_19.pt")
    #-starting_epoch = checkpoint_frontal['epoch'] + 1 
    
    
    #Initialize CNN for frontal images:
    model_frontal = CnnLstmModel(512, 3, 11, True, device1)
    #-model_frontal.load_state_dict(checkpoint_frontal['model_state_dict'])
    model_frontal.to(device1)
    
        
    optimizer_frontal = torch.optim.AdamW(model_frontal.parameters(), lr=0.00001, weight_decay=0.0) #decay: 0.001--> verschlechterung, 0.0001-> gleich,  lr=0.00001
    scheduler_frontal = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_frontal, 'min', factor=0.1, patience=10, verbose=True)
    loss_function_frontal_1 = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1))
    loss_function_frontal_2 = torch.nn.MSELoss()
       
    
    
    #Initialize CNN for lateral images:
    model_lateral = CnnLstmModel(512, 3, 11, True, device2)
    #-model_lateral.load_state_dict(checkpoint_lateral['model_state_dict'])
    model_lateral.to(device2)

    
    optimizer_lateral = torch.optim.AdamW(model_lateral.parameters(), lr=0.00001, weight_decay=0.0)
    scheduler_lateral = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_lateral, 'min', factor=0.1, patience=10, verbose=True)
    loss_function_lateral_1 = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1))
    loss_function_lateral_2 = torch.nn.MSELoss()
    
    
    
    
    best_loss_frontal = 0.0
    best_acc_frontal = 0.0
    running_loss_frontal = 0.0
    
    best_loss_lateral = 0.0
    best_acc_lateral = 0.0
    running_loss_lateral = 0.0
    
    
    loss_function_validation_1 = torch.nn.BCEWithLogitsLoss()
    loss_function_validation_2 = torch.nn.MSELoss()
    
    #-del checkpoint_frontal, checkpoint_lateral
    
    
    for epoch in range(500):
        model_frontal.train()
        model_lateral.train()
        model_frontal.requires_grad_(True)
        model_lateral.requires_grad_(True)
        
        modelEvaluationTrain.reset()
        
        '''
        count = 0
        for child in model_frontal.cnn.children():
            count += 1
            if count < 9:
                for param in child.parameters():
                    param.requires_grad = False
        count = 0
        for child in model_lateral.cnn.children():
            count += 1
            if count < 9:
                for param in child.parameters():
                    param.requires_grad = False
        '''
        
        
        #print("learning_rate = {}".format(scheduler_frontal.get_last_lr()))
        
        for step, batch in enumerate(dataLoaderTrain):
            
            #if step >= 5:
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
            
                    
            labels_frontal = hasThrombus_frontal.to(device=device1, dtype=torch.float)
            keypoints_frontal = batch['keypoints'].to(device=device1, dtype=torch.float)
            images_frontal = batch['image'].to(device=device1, dtype=torch.float)
            
            labels_lateral = hasThrombus_lateral.to(device=device2, dtype=torch.float)
            keypoints_lateral = batch['keypointsOtherView'].to(device=device2, dtype=torch.float)
            images_lateral = batch['imageOtherView'].to(device=device2, dtype=torch.float)
            
            keypoints_frontal = keypoints_frontal.squeeze(0)
            keypoints_lateral = keypoints_lateral.squeeze(0)
            
            
            optimizer_frontal.zero_grad()
            optimizer_lateral.zero_grad()
            
            output_frontal = model_frontal(images_frontal)
            output_lateral = model_lateral(images_lateral)
            
            #print(output_frontal)
            output_frontal_hasThrombus = output_frontal[0, 0]
            output_lateral_hasThrombus = output_lateral[0, 0]
            output_frontal_hasThrombus = output_frontal_hasThrombus.unsqueeze(0).unsqueeze(0)
            output_lateral_hasThrombus = output_lateral_hasThrombus.unsqueeze(0).unsqueeze(0)
            
            output_frontal_keypoints = output_frontal[0, 1:]
            output_lateral_keypoints = output_lateral[0, 1:]
            keypoints_frontal = keypoints_frontal.view(-1)
            keypoints_lateral = keypoints_lateral.view(-1)
            
            
            del images_frontal
            del images_lateral
            
            loss_frontal_1 = loss_function_frontal_1(output_frontal_hasThrombus, labels_frontal)
            loss_frontal_2 = loss_function_frontal_2(output_frontal_keypoints, keypoints_frontal)
            loss_frontal = loss_frontal_1 + loss_frontal_2
            
            loss_lateral_1 = loss_function_lateral_1(output_lateral_hasThrombus, labels_lateral)
            loss_lateral_2 = loss_function_lateral_2(output_lateral_keypoints, keypoints_lateral)
            loss_lateral = loss_lateral_1 + loss_lateral_2
            
            #print(loss_lateral_1)
            #print(loss_lateral_2)
            loss_frontal.backward()
            loss_lateral.backward()
            
            #print(model_lateral.weight.grad.abs())
            if epoch == 0 and (step == 1 or step == 15):
                plot_grad_flow(model_frontal.named_parameters())
            
            del labels_frontal
            del labels_lateral
            optimizer_frontal.step()
            optimizer_lateral.step()
            running_loss_frontal += loss_frontal.item()           
            running_loss_lateral += loss_lateral.item()
            

            #--scheduler_frontal.step()
            #--scheduler_lateral.step()
            
            #-------------- Evaluate Training ACC PREC and Recall ------------
            estimate_train_frontal = THROMBUS_NO if torch.sigmoid(output_frontal_hasThrombus).item() <= 0.5 else THROMBUS_YES
            estimate_train_lateral = THROMBUS_NO if torch.sigmoid(output_lateral_hasThrombus).item() <= 0.5 else THROMBUS_YES
            
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
            
        #------------- Ende for loop training ---------------------------------
        # ------------- Schedular Steps ---------------------------------------
        scheduler_frontal.step(running_loss_frontal / (step + 1))
        scheduler_lateral.step(running_loss_lateral / (step + 1))
        #------------- Print Loss statistics and save models ------------------
        print('Epoche {}'.format(epoch))
        print('loss_frontal = {} ; loss_lateral = {}'.format(running_loss_frontal / (step + 1), running_loss_lateral / (step + 1) ))
        modelEvaluationTrain.printAllStats()
        
        running_loss_frontal = 0.0
        running_loss_lateral = 0.0
        
        path_frontal = PATH + '_frontal_last' + '.pt'
        torch.save({
        'epoch': epoch,
        'model_state_dict': model_frontal.state_dict(),
        'optimizer_state_dict': optimizer_frontal.state_dict(),
        'loss': loss_frontal}, path_frontal)
        
        path_lateral = PATH + '_lateral_last' + '.pt'
        torch.save({
        'epoch': epoch,
        'model_state_dict': model_lateral.state_dict(),
        'optimizer_state_dict': optimizer_lateral.state_dict(),
        'loss': loss_lateral}, path_lateral)
        
        if epoch % 20 == 19:
            path_frontal = PATH + '_frontal_' + str(epoch) + '.pt'
            torch.save({
            'epoch': epoch,
            'model_state_dict': model_frontal.state_dict(),
            'optimizer_state_dict': optimizer_frontal.state_dict(),
            'loss': loss_frontal}, path_frontal)
            
            path_lateral = PATH + '_lateral_' + str(epoch) + '.pt'
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
            keypoints_frontal = batch['keypoints'].to(device=device1, dtype=torch.float)
            labels_lateral = hasThrombus_lateral.to(device=device2, dtype=torch.float)
            keypoints_lateral = batch['keypointsOtherView'].to(device=device2, dtype=torch.float)
            
            images_frontal = batch['image'].to(device=device1, dtype=torch.float)
            images_lateral = batch['imageOtherView'].to(device=device2, dtype=torch.float)

            keypoints_frontal = keypoints_frontal.squeeze(0)
            keypoints_lateral = keypoints_lateral.squeeze(0)
            keypoints_frontal = keypoints_frontal.view(-1)
            keypoints_lateral = keypoints_lateral.view(-1)

            output_frontal = model_frontal(images_frontal)
            output_lateral = model_lateral(images_lateral)

            output_frontal_hasThrombus = output_frontal[0, 0]
            output_lateral_hasThrombus = output_lateral[0, 0]
            output_frontal_hasThrombus = output_frontal_hasThrombus.unsqueeze(0).unsqueeze(0)
            output_lateral_hasThrombus = output_lateral_hasThrombus.unsqueeze(0).unsqueeze(0)
            
            output_frontal_keypoints = output_frontal[0, 1:]
            output_lateral_keypoints = output_lateral[0, 1:]
            
            if step == 1 or step == 2:
                print('GT')
                print(keypoints_lateral)
                print('Estimated')
                print(output_lateral_keypoints)

            del images_frontal
            del images_lateral
            
            validation_loss_frontal_1 = loss_function_validation_1(output_frontal_hasThrombus, labels_frontal)
            validation_loss_frontal_2 = loss_function_validation_2(output_frontal_keypoints, keypoints_frontal)
            validation_loss_frontal += validation_loss_frontal_1.item() + validation_loss_frontal_2.item()
            
            validation_loss_lateral_1 = loss_function_validation_1(output_lateral_hasThrombus, labels_lateral)
            validation_loss_lateral_2 = loss_function_validation_2(output_lateral_keypoints, keypoints_lateral)
            validation_loss_lateral += validation_loss_lateral_1.item() + validation_loss_lateral_2.item()
                      
            
            
            
            #------------- Evaluate Validation ACC PREC and Recall ------------
            estimate_frontal = THROMBUS_NO if torch.sigmoid(output_frontal_hasThrombus).item() <= 0.5 else THROMBUS_YES
            estimate_lateral = THROMBUS_NO if torch.sigmoid(output_lateral_hasThrombus).item() <= 0.5 else THROMBUS_YES
            
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

        
        if best_acc_frontal == 0:
            best_acc_frontal = modelEvaluationTest.getAccuracyFrontal()
            path_frontal = PATH + '_frontal_best_acc' + '.pt'
            torch.save({
            'epoch': epoch,
            'model_state_dict': model_frontal.state_dict(),
            'optimizer_state_dict': optimizer_frontal.state_dict(),
            'loss': validation_loss_frontal / (step + 1),
            'acc': best_acc_frontal}, path_frontal)
            
        elif best_acc_frontal < modelEvaluationTest.getAccuracyFrontal():
            best_acc_frontal = modelEvaluationTest.getAccuracyFrontal()
            path_frontal = PATH + '_frontal_best_acc' + '.pt'
            torch.save({
            'epoch': epoch,
            'model_state_dict': model_frontal.state_dict(),
            'optimizer_state_dict': optimizer_frontal.state_dict(),
            'loss': validation_loss_frontal / (step + 1),
            'acc': best_acc_frontal}, path_frontal)
            
        
        if best_acc_lateral == 0:
            best_acc_lateral = modelEvaluationTest.getAccuracyLateral()
            path_lateral = PATH + '_lateral_best_acc' + '.pt'
            torch.save({
            'epoch': epoch,
            'model_state_dict': model_lateral.state_dict(),
            'optimizer_state_dict': optimizer_lateral.state_dict(),
            'loss': validation_loss_lateral / (step + 1),
            'acc': best_acc_lateral}, path_lateral)
            
        elif best_acc_lateral < modelEvaluationTest.getAccuracyLateral():
            best_acc_lateral = modelEvaluationTest.getAccuracyLateral()
            path_lateral = PATH + '_lateral_best_acc' + '.pt'
            torch.save({
            'epoch': epoch,
            'model_state_dict': model_lateral.state_dict(),
            'optimizer_state_dict': optimizer_lateral.state_dict(),
            'loss': validation_loss_lateral / (step + 1),
            'acc': best_acc_lateral}, path_lateral)
        

        if best_loss_frontal == 0:
            best_loss_frontal = validation_loss_frontal / (step + 1)
            path_frontal = PATH + '_frontal_best_evalloss' + '.pt'
            torch.save({
            'epoch': epoch,
            'model_state_dict': model_frontal.state_dict(),
            'optimizer_state_dict': optimizer_frontal.state_dict(),
            'loss': best_loss_frontal}, path_frontal)
        
        elif best_loss_frontal > validation_loss_frontal / (step + 1):
            best_loss_frontal = validation_loss_frontal / (step + 1)
            path_frontal = PATH + '_frontal_best_evalloss' + '.pt'
            torch.save({
            'epoch': epoch,
            'model_state_dict': model_frontal.state_dict(),
            'optimizer_state_dict': optimizer_frontal.state_dict(),
            'loss': best_loss_frontal}, path_frontal)
        
    
        if best_loss_lateral == 0:
            best_loss_lateral = validation_loss_lateral / (step + 1)
            path_lateral = PATH + '_lateral_best_evalloss' + '.pt'
            torch.save({
            'epoch': epoch,
            'model_state_dict': model_lateral.state_dict(),
            'optimizer_state_dict': optimizer_lateral.state_dict(),
            'loss': best_loss_lateral}, path_lateral)
        
        elif best_loss_lateral > validation_loss_lateral / (step + 1):
            best_loss_lateral = validation_loss_lateral / (step + 1)
            path_lateral = PATH + '_lateral_best_evalloss' + '.pt'
            torch.save({
            'epoch': epoch,
            'model_state_dict': model_lateral.state_dict(),
            'optimizer_state_dict': optimizer_lateral.state_dict(),
            'loss': best_loss_lateral}, path_lateral)
        
    #pr.disable()
    #p = pstats.Stats(pr)
    #p.sort_stats(SortKey.TIME).print_stats(15) # Alternative: SortKey.CUMULATIVE
    