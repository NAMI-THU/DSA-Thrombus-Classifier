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
import torchvision.models as models
import torch.cuda
import torch.optim
import torch.nn
from torch import autograd

if __name__ == "__main__":
    
    #pr = cProfile.Profile()
    #pr.enable()

    torch.set_num_threads(8)    
    
    LOAD_PATH = "/media/nami/TE_GZ/Datenauswertung_DSA-Bilder/resnet101_adam-optim_ohne_affinetransf/"
    SAVE_PATH = "/media/nami/TE_GZ/Datenauswertung_DSA-Bilder/resnet101_finetuning_v2/"
    data_path = "/media/nami/TE_GZ/DSA-aufbereitet-nifti"
    #data_path = "C:\\Datasets\\Daten-Guenzburg\\Tests"
    #data_path = "C:\\Daten-Guenzburg\\nifti"

    #csv_path = "C:\\Daten-Guenzburg\\Datenauswertung_DSA-Bilder"
    csv_path = "/media/nami/TE_GZ/Datenauswertung_DSA-Bilder"

    modelEvaluation = ModelEvaluation()

    data_set_train = DsaDataset(data_path, csv_path, "Dataset_1_train.csv", training=True)
    data_set_train.loadCsvData()    
    data_set_train.createDatasetDict()
    #sample = data_set_train.__getitem__(0)

    data_set_test = DsaDataset(data_path, csv_path, "Dataset_1_test.csv", training=False)
    data_set_test.loadCsvData()
    data_set_test.createDatasetDict()

    batchSize = 2
    dataLoaderTrain = DataLoader(dataset=data_set_train, batch_size=batchSize, shuffle=True, sampler=None,
                            batch_sampler=None, num_workers=8, collate_fn=None,
                            pin_memory=False, drop_last=False, timeout=1200,
                            worker_init_fn=None)
    
    dataLoaderTest = DataLoader(dataset=data_set_test, batch_size=1, shuffle=False, sampler=None,
                            batch_sampler=None, num_workers=8, collate_fn=None,
                            pin_memory=False, drop_last=False, timeout=1200,
                            worker_init_fn=None)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #Load Checkpoints:
    checkpoint_frontal = torch.load(LOAD_PATH + "resnet101_frontal_129.pt")        
    checkpoint_lateral = torch.load(LOAD_PATH + "resnet101_lateral_129.pt")
    starting_epoch = checkpoint_frontal['epoch'] + 1 


    #Initialize CNN for frontal images:
    resnet18_frontal = models.resnet101()  
    resnet18_frontal.conv1 = torch.nn.Conv2d(62, 64, kernel_size=7, stride=2, padding=3, bias=False)
    resnet18_frontal.fc = torch.nn.Linear(resnet18_frontal.fc.in_features, 1)
    resnet18_frontal.load_state_dict(checkpoint_frontal['model_state_dict'])
    resnet18_frontal.to(device)
    
    optimizer_frontal = torch.optim.Adam(resnet18_frontal.parameters(), lr=0.000001)
    optimizer_frontal.load_state_dict(checkpoint_frontal['optimizer_state_dict'])
    scheduler_frontal = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_frontal, 'min', patience=5, factor=0.5, verbose=True)
    loss_function_frontal = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.54))
    

    running_loss_frontal = 0.0
    
    #Initialize CNN for lateral images:
    resnet18_lateral = models.resnet101()
    resnet18_lateral.conv1 = torch.nn.Conv2d(62, 64, kernel_size=7, stride=2, padding=3, bias=False)
    resnet18_lateral.fc = torch.nn.Linear(resnet18_lateral.fc.in_features, 1)
    resnet18_lateral.load_state_dict(checkpoint_lateral['model_state_dict'])
    resnet18_lateral.to(device)
    
    optimizer_lateral = torch.optim.Adam(resnet18_lateral.parameters(), lr=0.000001)
    optimizer_lateral.load_state_dict(checkpoint_lateral['optimizer_state_dict'])
    scheduler_lateral = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_frontal, 'min', patience=5, factor=0.5, verbose=True)
    loss_function_lateral = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.54))
    
    running_loss_lateral = 0.0
        
    loss_function_validation = torch.nn.BCEWithLogitsLoss()
    

    
    loss_container_frontal = list()
    loss_container_lateral = list()
    loss_container_validation_frontal = list()
    loss_container_validation_lateral = list()

    
    
    for epoch in range(starting_epoch, 140):
        resnet18_frontal.train()
        resnet18_lateral.train()
        resnet18_frontal.requires_grad_(True)
        resnet18_lateral.requires_grad_(True)
        
        for step, batch in enumerate(dataLoaderTrain):
            
            #if step >= 20:
            #    break
    
            hasThrombus_frontal = torch.zeros((batch['keypoints'].shape[0],1))
            for index1, keypoint1 in enumerate(batch['keypoints']):
                hasThrombus_frontal[index1] = 0 if torch.max(keypoint1) == 0 else 1
                
            hasThrombus_lateral = torch.zeros((batch['keypointsOtherView'].shape[0],1))
            for index2, keypoint2 in enumerate(batch['keypointsOtherView']):
                hasThrombus_lateral[index2] = 0 if torch.max(keypoint2) == 0 else 1
                # WICHTIG:
                # Beim Extraktionsprozess noch bedenken, dass eine Ansicht als Thrombusfrei
                # markiert ist, die andere Ansicht aber als nicht thrombusfrei 
                # Das muss hier überprüft und entsprechend behandelt werden bzgl.
                # Labelung !!!
                   
                    
            labels_frontal = hasThrombus_frontal.to(device=device, dtype=torch.float)
            images_frontal = batch['image'].to(device=device, dtype=torch.float)
            
            labels_lateral = hasThrombus_lateral.to(device=device, dtype=torch.float)
            images_lateral = batch['imageOtherView'].to(device=device, dtype=torch.float)
        
            #torch.autograd.set_detect_anomaly(True)
            optimizer_frontal.zero_grad()
            output_frontal = resnet18_frontal(images_frontal)
            del images_frontal
            loss_frontal = loss_function_frontal(output_frontal, labels_frontal)
            loss_frontal.backward()
            del labels_frontal
            optimizer_frontal.step()
            running_loss_frontal += loss_frontal.item()
            
            #==============================================================
            '''
            #Hier abänderung wegen gleiches Model Frontal  + Lateral:
            optimizer_frontal.zero_grad()
            output_lateral = resnet18_frontal(images_lateral)
            del images_lateral
            loss_lateral = loss_function_frontal(output_lateral, labels_lateral)
            loss_lateral.backward()
            del labels_lateral
            optimizer_frontal.step()
            running_loss_lateral += loss_lateral.item()
            '''
            #==============================================================
            
            
            optimizer_lateral.zero_grad()
            output_lateral = resnet18_lateral(images_lateral)
            del images_lateral
            loss_lateral = loss_function_lateral(output_lateral, labels_lateral)
            loss_lateral.backward()
            del labels_lateral
            optimizer_lateral.step()
            running_loss_lateral += loss_lateral.item()
            
            
            #if step % 8 == 7:
            #    print(step)

        #------------- Ende for loop training ---------------------------------
        #------------- Print Loss statistics and save models ------------------
        print('Epoche {}'.format(epoch))
        print('loss_frontal = {} ; loss_lateral = {}'.format(running_loss_frontal / (step + 1), running_loss_lateral / (step + 1) ))
        loss_container_frontal.append(running_loss_frontal / (step + 1))
        running_loss_frontal = 0.0
        loss_container_lateral.append(running_loss_lateral / (step + 1))
        running_loss_lateral = 0.0
                
        #if epoch % 5 == 4:
        path_frontal = SAVE_PATH + '_frontal_' + str(epoch) + '.pt'
        torch.save({
        'epoch': epoch,
        'model_state_dict': resnet18_frontal.state_dict(),
        'optimizer_state_dict': optimizer_frontal.state_dict(),
        'loss': loss_frontal}, path_frontal)
        
        path_lateral = SAVE_PATH + '_lateral_' + str(epoch) + '.pt'
        torch.save({
        'epoch': epoch,
        'model_state_dict': resnet18_lateral.state_dict(),
        'optimizer_state_dict': optimizer_lateral.state_dict(),
        'loss': loss_lateral}, path_lateral)
        
        #----------------- Evaluate Model -------------------------------------
        resnet18_frontal.eval()
        resnet18_lateral.eval()
        resnet18_frontal.requires_grad_(False)
        resnet18_lateral.requires_grad_(False)  
        
        modelEvaluation.reset()
        validation_loss_frontal = 0
        validation_loss_lateral = 0
        
        for step, batch in enumerate(dataLoaderTest):
            
            #if step >= 3:
            #    break
        
            hasThrombus_frontal = torch.tensor([[0.0]]) if torch.max(batch['keypoints']) == 0 else torch.tensor([[1.0]])      
            hasThrombus_lateral = torch.tensor([[0.0]]) if torch.max(batch['keypointsOtherView']) == 0 else torch.tensor([[1.0]])
            
            labels_frontal = hasThrombus_frontal.to(device=device, dtype=torch.float)
            labels_lateral = hasThrombus_lateral.to(device=device, dtype=torch.float)
            
            images_frontal = batch['image'].to(device=device, dtype=torch.float)
            images_lateral = batch['imageOtherView'].to(device=device, dtype=torch.float)

            output_frontal = resnet18_frontal(images_frontal)  
            #Hier abänderung lateral --> frontal wegen gleichem Netzwerk für Frontal + Lateral:
            output_lateral = resnet18_lateral(images_lateral)

            
            validation_loss_frontal += loss_function_validation(output_frontal, labels_frontal).item()
            validation_loss_lateral += loss_function_validation(output_lateral, labels_lateral).item()
            
            
            del images_frontal
            del images_lateral
            
            estimate_frontal = 0 if torch.sigmoid(output_frontal).item() <= 0.5 else 1
            estimate_lateral = 0 if torch.sigmoid(output_lateral).item() <= 0.5 else 1
            
            if estimate_frontal == 0:
                if torch.max(batch['keypoints']) == 0:
                    modelEvaluation.increaseTNfrontal()
                else:
                    modelEvaluation.increaseFNfrontal()
            else: # means: estimate_frontal = 1
                if torch.max(batch['keypoints']) == 0:
                    modelEvaluation.increaseFPfrontal()
                else:
                    modelEvaluation.increaseTPfrontal()
            
            if estimate_lateral == 0:
                if torch.max(batch['keypointsOtherView']) == 0:
                    modelEvaluation.increaseTNlateral()
                else:
                    modelEvaluation.increaseFNlateral()
            else: # means: estimate_lateral = 1
                if torch.max(batch['keypointsOtherView']) == 0:
                    modelEvaluation.increaseFPlateral()
                else:
                    modelEvaluation.increaseTPlateral()
                    
                  
        # ------------- Ende for loop validation data loader ------------------
        print('val_loss_frontal = {} ; val_loss_lateral = {}'.format(validation_loss_frontal / (step + 1), validation_loss_lateral / (step + 1)))
        loss_container_validation_frontal.append(validation_loss_frontal / (step + 1))
        loss_container_validation_lateral.append(validation_loss_lateral / (step + 1))
        modelEvaluation.printAllStats()

        # ------------- Schedular Steps ---------------------------------------
        #scheduler_frontal.step((validation_loss_frontal / (step + 1)) + (validation_loss_lateral / (step + 1)))
        #scheduler_lateral.step()


    print(loss_container_frontal)
    print(loss_container_lateral)
    print(loss_container_validation_frontal)
    print(loss_container_validation_lateral)
    #pr.disable()
    #p = pstats.Stats(pr)
    #p.sort_stats(SortKey.TIME).print_stats(15) # Alternative: SortKey.CUMULATIVE
    