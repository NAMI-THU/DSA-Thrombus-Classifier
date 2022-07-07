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
#import matplotlib
#matplotlib.use('tkagg')

import matplotlib.pyplot as plt

import numpy as np

from IndexTracker import IndexTracker

from object_locator.object_locator import losses
from object_locator.object_locator.models import unet_model
from object_locator.object_locator.metrics import Judge

THROMBUS_NO = 0.214
THROMBUS_YES = 0.786



if __name__ == "__main__":
    

    torch.set_num_threads(8)    
    
    PATH = "/media/nami/TE_GZ/Datenauswertung_DSA-Bilder/Trainings-loss-Testlaeufe/Unet_v4/"
    data_path = "/media/nami/TE_GZ/DSA-aufbereitet-nifti"
    #data_path = "C:\\Datasets\\Daten-Guenzburg\\Tests"
    #data_path = "C:\\Daten-Guenzburg\\nifti"

    #csv_path = "C:\\Daten-Guenzburg\\Datenauswertung_DSA-Bilder"
    csv_path = "/media/nami/TE_GZ/Datenauswertung_DSA-Bilder"

    modelEvaluation = ModelEvaluation()

    data_set_train = DsaDataset(data_path, csv_path, "Dataset_3_train.csv", training=False)
    data_set_train.loadCsvData()    
    data_set_train.createDatasetDict()

    data_set_test = DsaDataset(data_path, csv_path, "Dataset_3_test.csv", training=False)
    data_set_test.loadCsvData()
    data_set_test.createDatasetDict()

    batchSize = 1
    dataLoaderTrain = DataLoader(dataset=data_set_train, batch_size=batchSize, shuffle=False, sampler=None,
                            batch_sampler=None, num_workers=8, collate_fn=None,
                            pin_memory=False, drop_last=False, timeout=2400,
                            worker_init_fn=None)
    
    dataLoaderTest = DataLoader(dataset=data_set_test, batch_size=1, shuffle=False, sampler=None,
                            batch_sampler=None, num_workers=8, collate_fn=None,
                            pin_memory=False, drop_last=False, timeout=2400,
                            worker_init_fn=None)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    #Load Checkpoints:
    checkpoint_frontal = torch.load(PATH + "_frontal_best_eval.pt")        
    checkpoint_lateral = torch.load(PATH + "_lateral_best_eval.pt")
    starting_epoch = checkpoint_frontal['epoch'] + 1 
    
    print(checkpoint_frontal['epoch'])
    print(checkpoint_frontal['loss'])
    print(checkpoint_lateral['epoch'])
    print(checkpoint_lateral['loss'])
    
    
    #Initialize Unet for frontal images:
    model_frontal =  unet_model.UNet(62, 1, height=1024, width=1024, device=device)
    model_frontal.load_state_dict(checkpoint_frontal['model_state_dict'])
    model_frontal.to(device)
    
    loss_function_regress_frontal = torch.nn.MSELoss()
    loss_function_loc_frontal = losses.WeightedHausdorffDistance(resized_height=1024,
                                            resized_width=1024, p=-1, return_2_terms=True,
                                            device=device)
    
    #Initialize CNN for lateral images:
    model_lateral = unet_model.UNet(62, 1, height=1024, width=1024, device=device)
    model_lateral.load_state_dict(checkpoint_lateral['model_state_dict'])
    model_lateral.to(device)

    
    loss_function_regress_lateral =  torch.nn.MSELoss()
    loss_function_loc_lateral = losses.WeightedHausdorffDistance(resized_height=1024,
                                            resized_width=1024, p=-1, return_2_terms=True,
                                            device=device)
        
    running_loss_frontal = 0.0
    running_loss1_frontal = 0.0
    running_loss2_frontal = 0.0
    running_loss3_frontal = 0.0
    
    running_loss_lateral = 0.0
    running_loss1_lateral = 0.0
    running_loss2_lateral = 0.0
    running_loss3_lateral = 0.0
    
    
    loss_function_regress_validation = torch.nn.MSELoss()
    loss_funciton_loc_validation = losses.WeightedHausdorffDistance(resized_height=1024,
                                            resized_width=1024, p=-1, return_2_terms=True,
                                            device=device)
    
    del checkpoint_frontal, checkpoint_lateral
    
    

    model_frontal.eval()
    model_lateral.eval()
    model_frontal.requires_grad_(False)
    model_lateral.requires_grad_(False)
    
    
    for step, batch in enumerate(dataLoaderTest):
        
        #if step < 15:
        #    continue
        if step > 30:
            break
        ##print("step = {}".format(step))
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
        
        original_sizes_lateral = torch.stack(original_sizes_lateral).to(device=device, dtype=torch.float)  
        count_keypoints_lateral = torch.stack(count_keypoints_lateral).to(device=device, dtype=torch.float)  
        labels_lateral = keypoints_lateral#.to(device=device, dtype=torch.float)
        images_lateral = batch['imageOtherView'].to(device=device, dtype=torch.float)


 
        # Evaluate frontal 1 batch:
        estimation_maps_frontal, estimation_counts_frontal = model_frontal(images_frontal)
        loss1_frontal, loss2_frontal = loss_function_loc_frontal.forward(estimation_maps_frontal,
                                                                         labels_frontal,
                                                                         original_sizes_frontal)
        estimation_counts_frontal = estimation_counts_frontal.view(-1)
        count_keypoints_frontal = count_keypoints_frontal.view(-1)
        
        loss3_frontal = loss_function_regress_frontal(estimation_counts_frontal, count_keypoints_frontal)
        loss_frontal = loss1_frontal + loss2_frontal + loss3_frontal

        
        image_frontal = images_frontal.squeeze().to(torch.device('cpu')).numpy().transpose((1,2,0)).astype(np.float32)
        image_frontal *= batch['imageStd'][0].numpy()
        image_frontal += batch['imageMean'][0].numpy()
        label_frontal = batch['keypoints'].squeeze().to(torch.device('cpu')).numpy().astype(np.int16)
        
        estimation_map_frontal = estimation_maps_frontal[0].unsqueeze(0).to(torch.device('cpu')).numpy().transpose((1,2,0)).astype(np.float32)
               
        fig1, ax1 = plt.subplots(1, 1)
        tracker1 = IndexTracker(ax1, image_frontal, 'Image1', label_frontal)
        fig1.canvas.mpl_connect('scroll_event', tracker1.onscroll)
        plt.show()
        
        fig1_1, ax1_1 = plt.subplots(1, 1)
        est_map_frontal = ax1_1.imshow(estimation_map_frontal[:, :, 0], cmap='gray', vmin=0, vmax=estimation_map_frontal.max())
        ax1_1.invert_yaxis()
        est_map_frontal.set_data(estimation_map_frontal[:, :, 0])
        est_map_frontal.axes.figure.canvas.draw()
        plt.show()
        


        # Evaluate lateral 1 batch:
        estimation_maps_lateral, estimation_counts_lateral = model_lateral(images_lateral)
        loss1_lateral, loss2_lateral = loss_function_loc_lateral.forward(estimation_maps_lateral,
                                                                         labels_lateral,
                                                                         original_sizes_lateral)
        estimation_counts_lateral = estimation_counts_lateral.view(-1)
        count_keypoints_lateral = count_keypoints_lateral.view(-1)
        
        print(estimation_counts_lateral)
        
        loss3_lateral = loss_function_regress_lateral(estimation_counts_lateral, count_keypoints_lateral)
        loss_lateral = loss1_lateral + loss2_lateral + loss3_lateral

        '''
        image_lateral = images_lateral.squeeze().to(torch.device('cpu')).numpy().transpose((1,2,0)).astype(np.float32)
        image_lateral *= batch['imageOtherViewStd'][0].numpy()
        image_lateral += batch['imageOtherViewMean'][0].numpy()
        label_lateral = batch['keypointsOtherView'].squeeze().to(torch.device('cpu')).numpy().astype(np.int16)
        
        estimation_map_lateral = estimation_maps_lateral[0].unsqueeze(0).to(torch.device('cpu')).numpy().transpose((1,2,0)).astype(np.float32)
               
        fig2, ax2 = plt.subplots(1, 1)
        tracker2 = IndexTracker(ax2, image_lateral, 'Image1', label_lateral)
        fig2.canvas.mpl_connect('scroll_event', tracker2.onscroll)
        plt.show()
        
        fig2_1, ax2_1 = plt.subplots(1, 1)
        est_map_lateral = ax2_1.imshow(estimation_map_lateral[:, :, 0], cmap='gray', vmin=0, vmax=estimation_map_lateral.max())
        ax2_1.invert_yaxis()
        est_map_lateral.set_data(estimation_map_lateral[:, :, 0])
        est_map_lateral.axes.figure.canvas.draw()
        plt.show()
        '''


    #------------- Ende for loop training ---------------------------------

    #------------- Print Loss statistics and save models ------------------
    '''
    print('l_f = {} ; l1_f = {} ; l2_f = {} ; l3_f = {}'.format(running_loss_frontal / (step + 1),
                                                                running_loss1_frontal / (step + 1),
                                                                running_loss2_frontal / (step + 1),
                                                                running_loss3_frontal / (step + 1) ))
    print('l_l = {} ; l1_l = {} ; l2_l = {} ; l3_l = {}'.format(running_loss_lateral / (step + 1),
                                                                running_loss1_lateral / (step + 1),
                                                                running_loss2_lateral / (step + 1),
                                                                running_loss3_lateral / (step + 1) ))
    '''
            



