#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 11:33:23 2020

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
plt.use('TkAgg')
import numpy as np

from object_locator.object_locator import losses
from object_locator.object_locator.models import unet_model
from object_locator.object_locator.metrics import Judge

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
        if(p.requires_grad) and (p is not None) and ("bias" not in n):
            if p.grad is not None:
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                print(p.grad.abs().mean())
                #print(p.abs().mean())
    plt.pyplot.plot(ave_grads, alpha=0.3, color="b")
    plt.pyplot.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.pyplot.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.pyplot.xlim(xmin=0, xmax=len(ave_grads))
    plt.pyplot.xlabel("Layers")
    plt.pyplot.ylabel("average gradient")
    plt.pyplot.title("Gradient flow")
    plt.pyplot.grid(True)
    #plt.pyplot.show()

if __name__ == "__main__":
    

    torch.set_num_threads(8)    
    
    PATH = "/media/nami/TE_GZ/Datenauswertung_DSA-Bilder/Unet_v4/"
    data_path = "/media/nami/TE_GZ/DSA-aufbereitet-nifti"
    #data_path = "C:\\Datasets\\Daten-Guenzburg\\Tests"
    #data_path = "C:\\Daten-Guenzburg\\nifti"

    #csv_path = "C:\\Daten-Guenzburg\\Datenauswertung_DSA-Bilder"
    csv_path = "/media/nami/TE_GZ/Datenauswertung_DSA-Bilder"

    modelEvaluationTrain = ModelEvaluation()
    modelEvaluationTest= ModelEvaluation()

    data_set_train = DsaDataset(data_path, csv_path, "Dataset_3_train.csv", training=True)
    data_set_train.loadCsvData()    
    data_set_train.createDatasetDict()
    

    data_set_test = DsaDataset(data_path, csv_path, "Dataset_3_valid.csv", training=False)
    data_set_test.loadCsvData()
    data_set_test.createDatasetDict()


    batchSize = 4 
    dataLoaderTrain = DataLoader(dataset=data_set_train, batch_size=batchSize, shuffle=True, sampler=None,
                            batch_sampler=None, num_workers=4, collate_fn=None,
                            pin_memory=False, drop_last=False, timeout=2400,
                            worker_init_fn=None)
    
    dataLoaderTest = DataLoader(dataset=data_set_test, batch_size=1, shuffle=False, sampler=None,
                            batch_sampler=None, num_workers=4, collate_fn=None,
                            pin_memory=False, drop_last=False, timeout=2400,
                            worker_init_fn=None)


    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Load Checkpoints:
    #-checkpoint_frontal = torch.load(PATH + "_frontal_last.pt")        
    #-checkpoint_lateral = torch.load(PATH + "_lateral_last.pt")
    #-starting_epoch = checkpoint_frontal['epoch'] + 1 
    
    
    #Initialize Unet for frontal images:
    model_frontal =  unet_model.UNet(62, 1, height=1024, width=1024, device=device)
    model_frontal = torch.nn.Sequential(model_frontal)
    model_frontal = torch.nn.DataParallel(model_frontal)
    #-model_frontal.load_state_dict(checkpoint_frontal['model_state_dict'])
    model_frontal.to(device)
    
    optimizer_frontal = torch.optim.AdamW(model_frontal.parameters(), lr=1e-4, weight_decay=0.0) #decay: 0.001--> verschlechterung, 0.0001-> gleich,  lr=0.00001
    #optimizer_frontal = torch.optim.SGD(model_frontal.parameters(), momentum=0.9, lr=0.0000001, weight_decay=0.0) # 0.0001 zu groß
    #-optimizer_frontal.load_state_dict(checkpoint_frontal['optimizer_state_dict'])
    #--scheduler_frontal = torch.optim.lr_scheduler.MultiStepLR(optimizer_frontal, milestones=[1], gamma=0.1)
    #scheduler_frontal = torch.optim.lr_scheduler.CyclicLR(optimizer_frontal, base_lr=0.000000001, max_lr=0.00001, step_size_up=len(dataLoaderTrain) * 10, cycle_momentum=False) 
    loss_function_regress_frontal = torch.nn.MSELoss() #BCEWithLogitsLoss(pos_weight=torch.tensor(1))
    loss_function_loc_frontal = losses.WeightedHausdorffDistance(resized_height=1024,
                                            resized_width=1024, p=-1, return_2_terms=True,
                                            device=device)
    
    #Initialize CNN for lateral images:
    model_lateral = unet_model.UNet(62, 1, height=1024, width=1024, device=device)
    model_lateral = torch.nn.Sequential(model_lateral)
    model_lateral = torch.nn.DataParallel(model_lateral)
    #-model_lateral.load_state_dict(checkpoint_lateral['model_state_dict'])
    model_lateral.to(device)

    
    optimizer_lateral = torch.optim.AdamW(model_lateral.parameters(), lr=1e-4, weight_decay=0.0)
    #optimizer_lateral = torch.optim.SGD(model_lateral.parameters(), momentum=0.9, lr=0.0000001, weight_decay=0.0) 
    #-optimizer_lateral.load_state_dict(checkpoint_lateral['optimizer_state_dict'])
    #--scheduler_lateral = torch.optim.lr_scheduler.MultiStepLR(optimizer_lateral, milestones=[1], gamma=0.1)
    #scheduler_lateral = torch.optim.lr_scheduler.CyclicLR(optimizer_lateral, base_lr=0.000000001, max_lr=0.00001, step_size_up=len(dataLoaderTrain) * 10, cycle_momentum=False)
    loss_function_regress_lateral =  torch.nn.MSELoss() #BCEWithLogitsLoss(pos_weight=torch.tensor(1))
    loss_function_loc_lateral = losses.WeightedHausdorffDistance(resized_height=1024,
                                            resized_width=1024, p=-1, return_2_terms=True,
                                            device=device)
        
    best_loss_frontal = 0.0
    running_loss_frontal = 0.0
    running_loss1_frontal = 0.0
    running_loss2_frontal = 0.0
    running_loss3_frontal = 0.0
    
    best_loss_lateral = 0.0
    running_loss_lateral = 0.0
    running_loss1_lateral = 0.0
    running_loss2_lateral = 0.0
    running_loss3_lateral = 0.0
    
    #-checkpoint_acc_frontal = torch.load(PATH + "_frontal_best_eval.pt")        
    #-checkpoint_acc_lateral = torch.load(PATH + "_lateral_best_eval.pt")
    #-best_acc_frontal = checkpoint_acc_frontal['loss']
    #-best_acc_lateral = checkpoint_acc_lateral['loss']
    #-print(best_acc_frontal)
    #-print(best_acc_lateral)
    
    
    loss_function_regress_validation = torch.nn.MSELoss() #BCEWithLogitsLoss(pos_weight=torch.tensor(1))
    loss_function_loc_validation = losses.WeightedHausdorffDistance(resized_height=1024,
                                            resized_width=1024, p=-1, return_2_terms=True,
                                            device=device)
    
    #-del checkpoint_frontal, checkpoint_lateral #-, checkpoint_acc_frontal, checkpoint_acc_lateral
    
    
    #torch.autograd.set_detect_anomaly(True)
    
    for epoch in range(9000):
        model_frontal.train()
        model_lateral.train()
        model_frontal.requires_grad_(True)
        model_lateral.requires_grad_(True)
        
        modelEvaluationTrain.reset()
        

        #-print("learning_rate = {}".format(scheduler_frontal.get_last_lr()))
        
        for step, batch in enumerate(dataLoaderTrain):
            print(step)
            #if step >= 20:
            #    break
            ##print("step = {}".format(step))
            
            hasThrombus_frontal = torch.zeros((batch['keypoints'].shape[0],1))
            for index1, keypoint1 in enumerate(batch['keypoints']):
                hasThrombus_frontal[index1] = THROMBUS_NO if torch.max(keypoint1) == 0 else THROMBUS_YES
                
            hasThrombus_lateral = torch.zeros((batch['keypointsOtherView'].shape[0],1))
            for index2, keypoint2 in enumerate(batch['keypointsOtherView']):
                hasThrombus_lateral[index2] = THROMBUS_NO if torch.max(keypoint2) == 0 else THROMBUS_YES
                
                
            original_sizes_frontal = list()
            count_keypoints_frontal = list()
            keypoints_frontal = list()
            for index, keypoints in enumerate(batch['keypoints']):
                ##print("keypoints = {}".format(keypoints))
                original_sizes_frontal.append(torch.tensor([[1024, 1024]]))
                count_keypoints_frontal.append(torch.tensor([[0]]))
                keypoint_list_frontal = list()
                for keypoint in keypoints:
                    if torch.max(keypoint) > 0:
                        ##print(keypoint)
                        keypoint_list_frontal.append(keypoint)
                        count_keypoints_frontal[index] += 1
                        
                if len(keypoint_list_frontal) > 0:
                    keypoints_frontal.append(torch.stack(keypoint_list_frontal).to(device=device, dtype=torch.float))
                else:
                    keypoints_frontal.append(torch.Tensor((0)).to(device=device, dtype=torch.float))
                    ##print("no thrombus {}".format(keypoints_frontal))
            
            original_sizes_lateral = list()
            count_keypoints_lateral = list() 
            keypoints_lateral = list()
            for index, keypoints in enumerate(batch['keypointsOtherView']):
                original_sizes_lateral.append(torch.tensor([[1024, 1024]]))
                count_keypoints_lateral.append(torch.tensor([[0]]))
                keypoint_list_lateral = list()
                for keypoint in keypoints:
                    if torch.max(keypoint) > 0:
                        keypoint_list_lateral.append(keypoint)
                        count_keypoints_lateral[index] += 1
                if len(keypoint_list_lateral) > 0:
                    keypoints_lateral.append(torch.stack(keypoint_list_lateral).to(device=device, dtype=torch.float))
                else:
                    keypoints_lateral.append(torch.Tensor((0)).to(device=device, dtype=torch.float))
                # WICHTIG:
                # Beim Extraktionsprozess noch bedenken, dass eine Ansicht als Thrombusfrei
                # markiert ist, die andere Ansicht aber als nicht thrombusfrei 
                # Das muss hier überprüft und entsprechend behandelt werden bzgl.
                # Labelung !!!
                   
             
            original_sizes_frontal = torch.stack(original_sizes_frontal).to(device=device, dtype=torch.float)  
            count_keypoints_frontal = torch.stack(count_keypoints_frontal).to(device=device, dtype=torch.float)  
            labels_frontal = keypoints_frontal#.to(device=device, dtype=torch.float)
            images_frontal = batch['image'].to(device=device, dtype=torch.float)
            hasThrombus_frontal = hasThrombus_frontal.to(device=device, dtype=torch.float)
            
            original_sizes_lateral = torch.stack(original_sizes_lateral).to(device=device, dtype=torch.float)  
            count_keypoints_lateral = torch.stack(count_keypoints_lateral).to(device=device, dtype=torch.float)  
            labels_lateral = keypoints_lateral#.to(device=device, dtype=torch.float)
            images_lateral = batch['imageOtherView'].to(device=device, dtype=torch.float)
            hasThrombus_lateral = hasThrombus_lateral.to(device=device, dtype=torch.float)


 
            # Training frontal 1 batch:
            optimizer_frontal.zero_grad()
            estimation_maps_frontal, estimation_counts_frontal = model_frontal(images_frontal)
            del images_frontal
            loss1_frontal, loss2_frontal = loss_function_loc_frontal.forward(estimation_maps_frontal,
                                                                             labels_frontal,
                                                                             original_sizes_frontal)
            estimation_counts_frontal = estimation_counts_frontal.view(-1)
            count_keypoints_frontal = count_keypoints_frontal.view(-1)
            hasThrombus_frontal = hasThrombus_frontal.view(-1)
            
            #print(torch.sigmoid(estimation_counts_frontal))
            #print(hasThrombus_frontal)
            
            loss3_frontal = loss_function_regress_frontal(estimation_counts_frontal, count_keypoints_frontal) #hasThrombus_frontal)
            ##loss3_frontal *= 200 #lambdaa
            loss_frontal = loss1_frontal + loss2_frontal + loss3_frontal
            loss_frontal.backward()
            if epoch == 0 and (step == 1 or step == 15):
                plot_grad_flow(model_frontal.named_parameters())
            ##if step == 1 or step == 8 or step == 15 or step == 100:
            ##    print("est = {}; true = {}".format(estimation_counts_frontal, count_keypoints_frontal))
            #for name, param in model_frontal.named_parameters():
            #    if param.grad is not None:
            #        print(name, param.grad.abs().mean())

            del labels_frontal
            optimizer_frontal.step()
            running_loss_frontal += loss_frontal.item()
            running_loss1_frontal += loss1_frontal.item()
            running_loss2_frontal += loss2_frontal.item()
            running_loss3_frontal += loss3_frontal.item()
            #scheduler_frontal.step()
            #print('l_f = {} ; l1_f = {} ; l2_f = {} ; l3_f = {}'.format(loss_frontal.item(),
            #                                                        loss1_frontal,
            #                                                        loss2_frontal,
            #                                                        loss3_frontal ))
            
            # Training lateral 1 batch:
            optimizer_lateral.zero_grad()
            estimation_maps_lateral, estimation_counts_lateral = model_lateral(images_lateral)
            del images_lateral
            loss1_lateral, loss2_lateral = loss_function_loc_lateral.forward(estimation_maps_lateral,
                                                                             labels_lateral,
                                                                             original_sizes_lateral)
            estimation_counts_lateral = estimation_counts_lateral.view(-1)
            count_keypoints_lateral = count_keypoints_lateral.view(-1)
            hasThrombus_lateral = hasThrombus_lateral.view(-1)
            
            loss3_lateral = loss_function_regress_lateral(estimation_counts_lateral, count_keypoints_lateral) # hasThrombus_lateral) 
            ##loss3_lateral *= 200 #lambdaa
            loss_lateral = loss1_lateral + loss2_lateral + loss3_lateral
            loss_lateral.backward()
            
            del labels_lateral
            optimizer_lateral.step()
            running_loss_lateral += loss_lateral.item()
            running_loss1_lateral += loss1_lateral.item()
            running_loss2_lateral += loss2_lateral.item()
            running_loss3_lateral += loss3_lateral.item()
            #scheduler_lateral.step()
            
            #-------- Training acc | prec | recall -----------------------------
            '''
            for index, estimation_count_frontal in enumerate(estimation_counts_frontal):
                estimate_frontal = THROMBUS_NO if torch.sigmoid(estimation_count_frontal).item() <= 0.5 else THROMBUS_YES
                
                if estimate_frontal == THROMBUS_NO:
                    if count_keypoints_frontal[index].item() == 0:
                        modelEvaluationTrain.increaseTNfrontal()
                    else:
                        modelEvaluationTrain.increaseFNfrontal()
                else: # means: estimate_frontal = 1
                    if count_keypoints_frontal[index].item()  == 0:
                        modelEvaluationTrain.increaseFPfrontal()
                    else:
                        modelEvaluationTrain.increaseTPfrontal()
            

            for index, estimation_count_lateral in enumerate(estimation_counts_lateral):
                estimate_lateral = THROMBUS_NO if torch.sigmoid(estimation_count_lateral).item() <= 0.5 else THROMBUS_YES
                
                if estimate_lateral == THROMBUS_NO:
                    if count_keypoints_lateral[index].item() == 0:
                        modelEvaluationTrain.increaseTNlateral()
                    else:
                        modelEvaluationTrain.increaseFNlateral()
                else: # means: estimate_lateral = 1
                    if count_keypoints_lateral[index].item() == 0:
                        modelEvaluationTrain.increaseFPlateral()
                    else:
                        modelEvaluationTrain.increaseTPlateral()
            '''

        #------------- Ende for loop training ---------------------------------
        # ------------- Schedular Steps ---------------------------------------
        #-scheduler_frontal.step() # running_loss_frontal / (step + 1))
        #-scheduler_lateral.step() # running_loss_lateral / (step + 1))
        #------------- Print Loss statistics and save models ------------------
        print('Epoche {}'.format(epoch))
        print('l_f = {} ; l1_f = {} ; l2_f = {} ; l3_f = {}'.format(running_loss_frontal / (step + 1),
                                                                    running_loss1_frontal / (step + 1),
                                                                    running_loss2_frontal / (step + 1),
                                                                    running_loss3_frontal / (step + 1) ))
        print('l_l = {} ; l1_l = {} ; l2_l = {} ; l3_l = {}'.format(running_loss_lateral / (step + 1),
                                                                    running_loss1_lateral / (step + 1),
                                                                    running_loss2_lateral / (step + 1),
                                                                    running_loss3_lateral / (step + 1) ))
        #-modelEvaluationTrain.printAllStats()
        
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
        
            
        running_loss_frontal = 0.0
        running_loss1_frontal = 0.0
        running_loss2_frontal = 0.0
        running_loss3_frontal = 0.0
        
            
        running_loss_lateral = 0.0
        running_loss1_lateral = 0.0
        running_loss2_lateral = 0.0
        running_loss3_lateral = 0.0
                
        if epoch % 100 == 99:
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
        if epoch % 2 == 1:
        
            model_frontal.eval()
            model_lateral.eval()
            model_frontal.requires_grad_(False)
            model_lateral.requires_grad_(False)  
            
            modelEvaluationTest.reset()
            
            validation_loss_frontal = 0.0
            validation_loss_1_frontal = 0.0
            validation_loss_2_frontal = 0.0
            validation_loss_3_frontal = 0.0
            
            validation_loss_lateral = 0.0
            validation_loss_1_lateral = 0.0
            validation_loss_2_lateral = 0.0
            validation_loss_3_lateral = 0.0
    
            
            for step, batch in enumerate(dataLoaderTest):
                
                hasThrombus_frontal = torch.zeros((batch['keypoints'].shape[0],1))
                for index1, keypoint1 in enumerate(batch['keypoints']):
                    hasThrombus_frontal[index1] = THROMBUS_NO if torch.max(keypoint1) == 0 else THROMBUS_YES
                
                hasThrombus_lateral = torch.zeros((batch['keypointsOtherView'].shape[0],1))
                for index2, keypoint2 in enumerate(batch['keypointsOtherView']):
                    hasThrombus_lateral[index2] = THROMBUS_NO if torch.max(keypoint2) == 0 else THROMBUS_YES
                
                #if step >= 3:
                #    break
                original_sizes_frontal = list()
                count_keypoints_frontal = list()
                keypoints_frontal = list()
                for index, keypoints in enumerate(batch['keypoints']):
                    ##print("keypoints = {}".format(keypoints))
                    original_sizes_frontal.append(torch.tensor([[1024, 1024]]))
                    count_keypoints_frontal.append(torch.tensor([[0]]))
                    keypoint_list_frontal = list()
                    for keypoint in keypoints:
                        if torch.max(keypoint) > 0:
                            keypoint_list_frontal.append(keypoint)
                            count_keypoints_frontal[index] += 1
                            
                    if len(keypoint_list_frontal) > 0:
                        keypoints_frontal.append(torch.stack(keypoint_list_frontal).to(device=device, dtype=torch.float))
                    else:
                        keypoints_frontal.append(torch.Tensor((0)).to(device=device, dtype=torch.float))
                
                original_sizes_lateral = list()
                count_keypoints_lateral = list() 
                keypoints_lateral = list()
                for index, keypoints in enumerate(batch['keypointsOtherView']):
                    original_sizes_lateral.append(torch.tensor([[1024, 1024]]))
                    count_keypoints_lateral.append(torch.tensor([[0]]))
                    keypoint_list_lateral = list()
                    for keypoint in keypoints:
                        if torch.max(keypoint) > 0:
                            keypoint_list_lateral.append(keypoint)
                            count_keypoints_lateral[index] += 1
                    if len(keypoint_list_lateral) > 0:
                        keypoints_lateral.append(torch.stack(keypoint_list_lateral).to(device=device, dtype=torch.float))
                    else:
                        keypoints_lateral.append(torch.Tensor((0)).to(device=device, dtype=torch.float))
                    # WICHTIG:
                    # Beim Extraktionsprozess noch bedenken, dass eine Ansicht als Thrombusfrei
                    # markiert ist, die andere Ansicht aber als nicht thrombusfrei 
                    # Das muss hier überprüft und entsprechend behandelt werden bzgl.
                    # Labelung !!!
                       
                 
                original_sizes_frontal = torch.stack(original_sizes_frontal).to(device=device, dtype=torch.float)  
                count_keypoints_frontal = torch.stack(count_keypoints_frontal).to(device=device, dtype=torch.float)  
                labels_frontal = keypoints_frontal#.to(device=device, dtype=torch.float)
                images_frontal = batch['image'].to(device=device, dtype=torch.float)
                hasThrombus_frontal = hasThrombus_frontal.to(device=device, dtype=torch.float)
                
                original_sizes_lateral = torch.stack(original_sizes_lateral).to(device=device, dtype=torch.float)  
                count_keypoints_lateral = torch.stack(count_keypoints_lateral).to(device=device, dtype=torch.float)  
                labels_lateral = keypoints_lateral#.to(device=device, dtype=torch.float)
                images_lateral = batch['imageOtherView'].to(device=device, dtype=torch.float)
                hasThrombus_lateral = hasThrombus_lateral.to(device=device, dtype=torch.float)
    
     
                # Validate frontal 1 batch:
                estimation_maps_frontal, estimation_counts_frontal = model_frontal(images_frontal)
                del images_frontal
                loss1_frontal, loss2_frontal = loss_function_loc_validation.forward(estimation_maps_frontal,
                                                                                    labels_frontal,
                                                                                    original_sizes_frontal)
                estimation_counts_frontal = estimation_counts_frontal.view(-1)
                count_keypoints_frontal = count_keypoints_frontal.view(-1)
                hasThrombus_frontal = hasThrombus_frontal.view(-1)
                
                loss3_frontal = loss_function_regress_validation(estimation_counts_frontal, count_keypoints_frontal) # hasThrombus_frontal)
                ##loss3_frontal *= 200 #lambdaa
                loss_frontal = loss1_frontal + loss2_frontal + loss3_frontal
    
                del labels_frontal
               
                validation_loss_frontal += loss_frontal.item()
                validation_loss_1_frontal += loss1_frontal.item()
                validation_loss_2_frontal += loss2_frontal.item()
                validation_loss_3_frontal += loss3_frontal.item()
                
                
                # Validate lateral 1 batch:
                estimation_maps_lateral, estimation_counts_lateral = model_lateral(images_lateral)
                del images_lateral
                loss1_lateral, loss2_lateral = loss_function_loc_validation.forward(estimation_maps_lateral,
                                                                                 labels_lateral,
                                                                                 original_sizes_lateral)
                estimation_counts_lateral = estimation_counts_lateral.view(-1)
                count_keypoints_lateral = count_keypoints_lateral.view(-1)
                hasThrombus_lateral = hasThrombus_lateral.view(-1)
                
                loss3_lateral = loss_function_regress_validation(estimation_counts_lateral, count_keypoints_lateral) #hasThrombus_lateral) 
                ##loss3_lateral *= 200 #lambdaa
                loss_lateral = loss1_lateral + loss2_lateral + loss3_lateral
                
                del labels_lateral
                validation_loss_lateral += loss_lateral.item()
                validation_loss_1_lateral += loss1_lateral.item()
                validation_loss_2_lateral += loss2_lateral.item()
                validation_loss_3_lateral += loss3_lateral.item()
               
                #-------- Validation acc | prec | recall -----------------------------
                '''
                for index, estimation_count_frontal in enumerate(estimation_counts_frontal):
                    estimate_frontal = THROMBUS_NO if torch.sigmoid(estimation_count_frontal).item() <= 0.5 else THROMBUS_YES
                    
                    if estimate_frontal == THROMBUS_NO:
                        if count_keypoints_frontal[index].item() == 0:
                            modelEvaluationTest.increaseTNfrontal()
                        else:
                            modelEvaluationTest.increaseFNfrontal()
                    else: # means: estimate_frontal = 1
                        if count_keypoints_frontal[index].item()  == 0:
                            modelEvaluationTest.increaseFPfrontal()
                        else:
                            modelEvaluationTest.increaseTPfrontal()
                
    
                for index, estimation_count_lateral in enumerate(estimation_counts_lateral):
                    estimate_lateral = THROMBUS_NO if torch.sigmoid(estimation_count_lateral).item() <= 0.5 else THROMBUS_YES
                    
                    if estimate_lateral == THROMBUS_NO:
                        if count_keypoints_lateral[index].item() == 0:
                            modelEvaluationTest.increaseTNlateral()
                        else:
                            modelEvaluationTest.increaseFNlateral()
                    else: # means: estimate_lateral = 1
                        if count_keypoints_lateral[index].item() == 0:
                            modelEvaluationTest.increaseFPlateral()
                        else:
                            modelEvaluationTest.increaseTPlateral()
                '''
                     
            # ------------- Ende for loop validation data loader ------------------
            print('val_l_f = {} ; val_l1_f = {} ; val_l2_f = {} ; val_l3_f = {}'.format(validation_loss_frontal / (step + 1),
                                                                        validation_loss_1_frontal / (step + 1),
                                                                        validation_loss_2_frontal / (step + 1),
                                                                        validation_loss_3_frontal / (step + 1) ))
            print('val_l_l = {} ; val_l1_l = {} ; val_l2_l = {} ; val_l3_l = {}'.format(validation_loss_lateral / (step + 1),
                                                                        validation_loss_1_lateral / (step + 1),
                                                                        validation_loss_2_lateral / (step + 1),
                                                                        validation_loss_3_lateral / (step + 1) ))
            #-modelEvaluationTest.printAllStats()
            '''
            if best_acc_frontal == 0:
                best_acc_frontal = modelEvaluationTest.getAccuracyFrontal()
                path_frontal = PATH + '_frontal_best_acc' + '.pt'
                torch.save({
                'epoch': epoch,
                'model_state_dict': model_frontal.state_dict(),
                'optimizer_state_dict': optimizer_frontal.state_dict(),
                'loss': loss_frontal,
                'acc': best_acc_frontal}, path_frontal)
                
            elif best_acc_frontal < modelEvaluationTest.getAccuracyFrontal():
                best_acc_frontal = modelEvaluationTest.getAccuracyFrontal()
                path_frontal = PATH + '_frontal_best_acc' + '.pt'
                torch.save({
                'epoch': epoch,
                'model_state_dict': model_frontal.state_dict(),
                'optimizer_state_dict': optimizer_frontal.state_dict(),
                'loss': loss_frontal,
                'acc': best_acc_frontal}, path_frontal)
                
            
            if best_acc_lateral == 0:
                best_acc_lateral = modelEvaluationTest.getAccuracyLateral()
                path_lateral = PATH + '_lateral_best_acc' + '.pt'
                torch.save({
                'epoch': epoch,
                'model_state_dict': model_lateral.state_dict(),
                'optimizer_state_dict': optimizer_lateral.state_dict(),
                'loss': loss_lateral,
                'acc': best_acc_lateral}, path_lateral)
                
            elif best_acc_lateral < modelEvaluationTest.getAccuracyLateral():
                best_acc_lateral = modelEvaluationTest.getAccuracyLateral()
                path_lateral = PATH + '_lateral_best_acc' + '.pt'
                torch.save({
                'epoch': epoch,
                'model_state_dict': model_lateral.state_dict(),
                'optimizer_state_dict': optimizer_lateral.state_dict(),
                'loss': loss_lateral,
                'acc': best_acc_lateral}, path_lateral)
            '''

            if best_loss_frontal == 0:
                best_loss_frontal = validation_loss_frontal / (step + 1)
                path_frontal = PATH + '_frontal_best_eval' + '.pt'
                torch.save({
                'epoch': epoch,
                'model_state_dict': model_frontal.state_dict(),
                'optimizer_state_dict': optimizer_frontal.state_dict(),
                'loss': best_loss_frontal}, path_frontal)
            
            elif best_loss_frontal > validation_loss_frontal / (step + 1):
                best_loss_frontal = validation_loss_frontal / (step + 1)
                path_frontal = PATH + '_frontal_best_eval' + '.pt'
                torch.save({
                'epoch': epoch,
                'model_state_dict': model_frontal.state_dict(),
                'optimizer_state_dict': optimizer_frontal.state_dict(),
                'loss': best_loss_frontal}, path_frontal)
            
        
            if best_loss_lateral == 0:
                best_loss_lateral = validation_loss_lateral / (step + 1)
                path_lateral = PATH + '_lateral_best_eval' + '.pt'
                torch.save({
                'epoch': epoch,
                'model_state_dict': model_lateral.state_dict(),
                'optimizer_state_dict': optimizer_lateral.state_dict(),
                'loss': best_loss_lateral}, path_lateral)
            
            elif best_loss_lateral > validation_loss_lateral / (step + 1):
                best_loss_lateral = validation_loss_lateral / (step + 1)
                path_lateral = PATH + '_lateral_best_eval' + '.pt'
                torch.save({
                'epoch': epoch,
                'model_state_dict': model_lateral.state_dict(),
                'optimizer_state_dict': optimizer_lateral.state_dict(),
                'loss': best_loss_lateral}, path_lateral)

        

    