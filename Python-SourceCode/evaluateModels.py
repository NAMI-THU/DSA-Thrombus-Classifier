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
from LSTMModel import LSTMModel
from CnnLstmModel import CnnLstmModel
import torchvision.models as models
import torch.cuda
import torch.optim
import torch.nn
from torch import autograd


THROMBUS_NO = 0.214
THROMBUS_YES = 0.786

if __name__ == "__main__":
    
    torch.set_num_threads(8)    
    
    PATH = "/media/nami/TE_GZ/Datenauswertung_DSA-Bilder/CrossFoldValidation/Models/CnnLstm/Regnet_y_16gf_Gru_resize/"
    data_path = "/media/nami/TE_GZ/DSA-aufbereitet-nifti"
    csv_path = "/media/nami/TE_GZ/Datenauswertung_DSA-Bilder/CrossFoldValidation/Datasets/"
    print(PATH)

    
    for fold in range(1, 6):
        torch.cuda.empty_cache()

        print('Fold {}'.format(fold))
        
        modelEvaluationTrain = ModelEvaluation()
        modelEvaluationTest = ModelEvaluation()  
    
        #data_set_test = DsaDataset(data_path, csv_path, "Dataset_4_fold" + str(fold) + "_valid.csv", training=False)
        data_set_test = DsaDataset(data_path, csv_path, "Dataset_4_test.csv", training=False)
        data_set_test.loadCsvData()
        data_set_test.createDatasetDict()
    
        batchSize = 1
        
        dataLoaderTest = DataLoader(dataset=data_set_test, batch_size=1, shuffle=False, sampler=None,
                                batch_sampler=None, num_workers=4, collate_fn=None,
                                pin_memory=False, drop_last=False, #timeout=2400,
                                worker_init_fn=None)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
        #Load Checkpoints:
        checkpoint_frontal = torch.load(PATH + "frontal_best_mcc_fold_" + str(fold) + ".pt")        
        checkpoint_lateral = torch.load(PATH + "lateral_best_mcc_fold_" + str(fold) + ".pt")
        epoch = checkpoint_frontal['epoch']
    
    
        #Initialize model for frontal images:
        #model_frontal = models.resnet152()  
        #model_frontal.conv1 = torch.nn.Conv2d(62, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #model_frontal.fc = torch.nn.Linear(model_frontal.fc.in_features, 1)
        #model_frontal = LSTMModel(1024*1024, 50, 2, 1, True)
        model_frontal = CnnLstmModel(512, 3, 1, True, device)
        #model_frontal = torch.nn.DataParallel(model_frontal)
        model_frontal.load_state_dict(checkpoint_frontal['model_state_dict'])
        model_frontal.to(device)
            
        
        #Initialize model for lateral images:
        #model_lateral = models.resnet152()
        #model_lateral.conv1 = torch.nn.Conv2d(62, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #model_lateral.fc = torch.nn.Linear(model_lateral.fc.in_features, 1)
        #model_lateral = LSTMModel(1024*1024, 50, 2, 1, True)
        model_lateral = CnnLstmModel(512, 3, 1, True, device)
        #model_lateral = torch.nn.DataParallel(model_lateral)
        model_lateral.load_state_dict(checkpoint_lateral['model_state_dict'])
        model_lateral.to(device)
        
    
        #----------------- Evaluate Models -------------------------------------
        model_frontal.eval()
        model_lateral.eval()
        model_frontal.requires_grad_(False)
        model_lateral.requires_grad_(False)  
            
        modelEvaluationTest.reset()
            
        estimates_probability_frontal = list()
        estimates_probability_lateral = list()
        estimates_thrombus_frontal = list()
        estimates_thrombus_lateral = list()
        
        gtruth_frontal = list()
        gtruth_lateral = list()
        
        
        for step, batch in enumerate(dataLoaderTest):
                    
            hasThrombus_frontal = torch.tensor([[THROMBUS_NO]]) if torch.max(batch['keypoints']) == 0 else torch.tensor([[THROMBUS_YES]])      
            hasThrombus_lateral = torch.tensor([[THROMBUS_NO]]) if torch.max(batch['keypointsOtherView']) == 0 else torch.tensor([[THROMBUS_YES]])
                
            for index in range(len(hasThrombus_frontal)):
                if not torch.equal(hasThrombus_frontal[index], hasThrombus_lateral[index]):
                    hasThrombus_frontal[index] = THROMBUS_YES
                    hasThrombus_lateral[index] = THROMBUS_YES
                    
            labels_frontal = hasThrombus_frontal.to(device=device, dtype=torch.float)
            labels_lateral = hasThrombus_lateral.to(device=device, dtype=torch.float)
            
            #For CNN or CNN + LSTM:
            images_frontal = batch['image']#.to(device=device, dtype=torch.float)
            images_lateral = batch['imageOtherView']#.to(device=device, dtype=torch.float)
    
            #For LSTM alone:
            #length_frontal = batch['image'].shape[1]
            #images_frontal = batch['image'].view(-1, length_frontal, 1024*1024)
            #images_frontal = images_frontal.to(device=device, dtype=torch.float)
            #length_lateral = batch['imageOtherView'].shape[1]
            #images_lateral = batch['imageOtherView'].view(-1, length_lateral, 1024*1024)
            #images_lateral = images_lateral.to(device=device, dtype=torch.float)
    
    
            output_frontal = model_frontal(images_frontal)
            output_lateral = model_lateral(images_lateral)
    
            
            #validation_loss_frontal += loss_function_validation(output_frontal, labels_frontal).item()
            #validation_loss_lateral += loss_function_validation(output_lateral, labels_lateral).item()
            
            
            del images_frontal
            del images_lateral
            
            
            #------------- Evaluate Validation ACC PREC and Recall ------------
            estimate_frontal = THROMBUS_NO if torch.sigmoid(output_frontal).item() <= 0.5 else THROMBUS_YES
            estimate_lateral = THROMBUS_NO if torch.sigmoid(output_lateral).item() <= 0.5 else THROMBUS_YES
            
            gtruth_frontal.append(hasThrombus_frontal)
            gtruth_lateral.append(hasThrombus_lateral)
            estimates_thrombus_frontal.append(estimate_frontal)
            estimates_thrombus_lateral.append(estimate_lateral)
            estimates_probability_frontal.append(torch.sigmoid(output_frontal).item())
            estimates_probability_lateral.append(torch.sigmoid(output_lateral).item())
    
                
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
            
            '''#Evaluate Accuracy, precision and recall together for lateral and frontal model
            compound_estimate = estimate_frontal | estimate_lateral
            
            if compound_estimate == 0:
                if torch.max(batch['keypoints']) == 0 and torch.max(batch['keypointsOtherView']) == 0:
                    compoundModelEvaluation.increaseTNfrontal()
                else:
                    compoundModelEvaluation.increaseFNfrontal()
            else: # means: compound_estimate = 1
                if torch.max(batch['keypoints']) == 0 and torch.max(batch['keypointsOtherView']) == 0:
                    compoundModelEvaluation.increaseFPfrontal()
                else:
                    compoundModelEvaluation.increaseTPfrontal()
            '''
                      
        # ------------- Ende for loop validation data loader ------------------
        '''
        print("Frontal")
        print(gtruth_frontal)
        print(estimates_probability_frontal)
        print(estimates_thrombus_frontal)
        print("Lateral")
        print(gtruth_lateral)
        print(estimates_probability_lateral)
        print(estimates_thrombus_lateral)
        '''
        
        modelEvaluationTest.printAllStats()
        #compoundModelEvaluation.printAllStats()
        