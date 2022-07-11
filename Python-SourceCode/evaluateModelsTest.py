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
import numpy as np


THROMBUS_NO = 0.214
THROMBUS_YES = 0.786
DATASET_LENGTH = 5

if __name__ == "__main__":
    
    torch.set_num_threads(8)    
    
    PATH = "/media/nami/TE_GZ/Datenauswertung_DSA-Bilder/CrossFoldValidation/Models/CnnLstm/Efficientnetv2_rw_s_Gru_resize/"
    data_path = "/media/nami/TE_GZ/DSA-aufbereitet-nifti"
    csv_path = "/media/nami/TE_GZ/Datenauswertung_DSA-Bilder/CrossFoldValidation/Datasets/"
    print(PATH)

    #data_set_test = DsaDataset(data_path, csv_path, "Dataset_4_fold" + str(fold) + "_valid.csv", training=False)
    data_set_test = DsaDataset(data_path, csv_path, "Dataset_thrombus_uebersehen.csv", training=False)
    data_set_test.loadCsvData()
    data_set_test.createDatasetDict()
    #print(data_set_test.__len__())
    
    all_estimates_probabilities_frontal = np.zeros((5, DATASET_LENGTH))
    all_estimates_probabilities_lateral = np.zeros((5, DATASET_LENGTH))
    
    gtruth_frontal = np.zeros((1, DATASET_LENGTH))
    gtruth_lateral = np.zeros((1, DATASET_LENGTH))

    for fold in range(1, 6):
        torch.cuda.empty_cache()

        print('Fold {}'.format(fold))
        
        modelEvaluationTrain = ModelEvaluation()
        modelEvaluationTest = ModelEvaluation()  
        compoundModelEvaluation = ModelEvaluation(compound=True)

        #data_set_test = DsaDataset(data_path, csv_path, "Dataset_4_fold" + str(fold) + "_valid.csv", training=False)
        #data_set_test = DsaDataset(data_path, csv_path, "Dataset_4_test.csv", training=False)
        #data_set_test.loadCsvData()
        #data_set_test.createDatasetDict()
    
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
        #print(epoch)
    
    
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
            
        
            
        #estimates_probability_frontal = list()
        #estimates_probability_lateral = list()
        estimates_thrombus_frontal = list()
        estimates_thrombus_lateral = list()
        
        mean_probability_frontal = 0.0
        mean_probability_lateral = 0.0
        
        
        for i in range(1):
            #print(i)
            modelEvaluationTest.reset()
            compoundModelEvaluation.reset()
            
            for step, batch in enumerate(dataLoaderTest):
                        
                hasThrombus_frontal = torch.tensor([[THROMBUS_NO]]) if torch.max(batch['keypoints']) == 0 else torch.tensor([[THROMBUS_YES]])      
                hasThrombus_lateral = torch.tensor([[THROMBUS_NO]]) if torch.max(batch['keypointsOtherView']) == 0 else torch.tensor([[THROMBUS_YES]])
                
                if fold == 1:
                    gtruth_frontal[0][step] = THROMBUS_NO if torch.max(batch['keypoints']) == 0 else THROMBUS_YES
                    gtruth_lateral[0][step] = THROMBUS_NO if torch.max(batch['keypointsOtherView']) == 0 else THROMBUS_YES
                    
                for index in range(len(hasThrombus_frontal)):
                    if not torch.equal(hasThrombus_frontal[index], hasThrombus_lateral[index]):
                        hasThrombus_frontal[index] = THROMBUS_YES
                        hasThrombus_lateral[index] = THROMBUS_YES
                        
                        if fold == 1:
                            gtruth_frontal[0][step] = THROMBUS_YES
                            gtruth_lateral[0][step] = THROMBUS_YES
                        
                labels_frontal = hasThrombus_frontal.to(device=device, dtype=torch.float)
                labels_lateral = hasThrombus_lateral.to(device=device, dtype=torch.float)
                
                #For CNN or CNN + LSTM:
                images_frontal = batch['image']#.to(device=device, dtype=torch.float)
                images_lateral = batch['imageOtherView']#.to(device=device, dtype=torch.float)

    

                output_frontal = model_frontal(images_frontal)
                output_lateral = model_lateral(images_lateral)
                
                
                del images_frontal
                del images_lateral
                
                
                #------------- Evaluate Validation ACC PREC and Recall ------------
                estimate_frontal = THROMBUS_NO if torch.sigmoid(output_frontal).item() <= 0.5 else THROMBUS_YES
                estimate_lateral = THROMBUS_NO if torch.sigmoid(output_lateral).item() <= 0.5 else THROMBUS_YES
                
                
                all_estimates_probabilities_frontal[fold - 1][step] += torch.sigmoid(output_frontal).item()
                all_estimates_probabilities_lateral[fold - 1][step] += torch.sigmoid(output_lateral).item()
                
                              
                
                # diese auswertung nun korrekt im Gegensatz zu "torch.max(batch['keypoints']) == 0"
                if estimate_frontal == THROMBUS_NO:
                    if gtruth_frontal[0][step] == THROMBUS_NO:
                        modelEvaluationTest.increaseTNfrontal()
                    else:
                        modelEvaluationTest.increaseFNfrontal()
                else: # means: estimate_frontal = 1
                    if gtruth_frontal[0][step] == THROMBUS_NO:
                        modelEvaluationTest.increaseFPfrontal()
                    else:
                        modelEvaluationTest.increaseTPfrontal()
                
                if estimate_lateral == THROMBUS_NO:
                    if gtruth_lateral[0][step] == THROMBUS_NO:
                        modelEvaluationTest.increaseTNlateral()
                    else:
                        modelEvaluationTest.increaseFNlateral()
                else: # means: estimate_lateral = 1
                    if gtruth_lateral[0][step] == THROMBUS_NO:
                        modelEvaluationTest.increaseFPlateral()
                    else:
                        modelEvaluationTest.increaseTPlateral()
                        
                '''    
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
                '''
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
            
            #print(all_estimates_probabilities_frontal)
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
    # ---------------------- Ende for loop folds ------------------------------
    
    ##all_estimates_probabilities_frontal /= 5
    ##all_estimates_probabilities_lateral /= 5
 
    
    mean_estimates_probabilities_frontal = np.sum(all_estimates_probabilities_frontal, axis=0) / 5
    mean_estimates_probabilities_lateral = np.sum(all_estimates_probabilities_lateral, axis=0) / 5
    
    compound_estimates_probabilities = (mean_estimates_probabilities_frontal + mean_estimates_probabilities_lateral) / 2
    
    modelEvaluationTest.reset()
    compoundModelEvaluation.reset()
    
    for index in range(DATASET_LENGTH):
        estimate_frontal = THROMBUS_NO if mean_estimates_probabilities_frontal[index] <= 0.5 else THROMBUS_YES
        estimate_lateral = THROMBUS_NO if mean_estimates_probabilities_lateral[index] <= 0.5 else THROMBUS_YES
        compound_estimate = THROMBUS_NO if compound_estimates_probabilities[index] <= 0.5 else THROMBUS_YES
        
        # Auswertung separat je Ansicht:
        if estimate_frontal == THROMBUS_NO:
            if gtruth_frontal[0][index] == THROMBUS_NO:
                modelEvaluationTest.increaseTNfrontal()
            else:
                modelEvaluationTest.increaseFNfrontal()
        else: # means: estimate_frontal = 1
            if gtruth_frontal[0][index] == THROMBUS_NO:
                modelEvaluationTest.increaseFPfrontal()
            else:
                modelEvaluationTest.increaseTPfrontal()
        
        if estimate_lateral == THROMBUS_NO:
            if gtruth_lateral[0][index] == THROMBUS_NO:
                modelEvaluationTest.increaseTNlateral()
            else:
                modelEvaluationTest.increaseFNlateral()
        else: # means: estimate_lateral = 1
            if gtruth_lateral[0][index] == THROMBUS_NO:
                modelEvaluationTest.increaseFPlateral()
            else:
                modelEvaluationTest.increaseTPlateral()
        
        # Auswertung für beide Ansichten gemeinsam:
        if compound_estimate == THROMBUS_NO:
            if gtruth_frontal[0][index] == THROMBUS_NO:
                compoundModelEvaluation.increaseTNfrontal()
            else:
                compoundModelEvaluation.increaseFNfrontal()
        else: # means: compound_estimate = THROMBUS_YES
            if gtruth_frontal[0][index] == THROMBUS_NO:
                compoundModelEvaluation.increaseFPfrontal()
            else:
                compoundModelEvaluation.increaseTPfrontal()
    
    print("Frontal GT:")
    print(gtruth_frontal)
    print("Frontal mean estimates:")
    print(mean_estimates_probabilities_frontal)
    
    print("Lateral GT:")
    print(gtruth_lateral)
    print("Lateral mean estimates:")
    print(mean_estimates_probabilities_lateral)
    
    print("Separate Model Evaluation:")
    modelEvaluationTest.printAllStats()
    
    print("Compound Model Evaluation:")
    compoundModelEvaluation.printAllStats()
        