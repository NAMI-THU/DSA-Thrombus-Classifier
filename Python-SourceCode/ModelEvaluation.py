#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 17:15:34 2020

@author: nami
"""

import math

class ModelEvaluation(object):
    def __init__(self, compound=False):
            
        self.TP_frontal = 0
        self.FP_frontal = 0
        self.TN_frontal = 0
        self.FN_frontal = 0
        
        self.TP_lateral = 0
        self.FP_lateral = 0
        self.TN_lateral = 0
        self.FN_lateral = 0
        self.compound = compound
        
        
    '''====================================================================='''
    def reset(self):
        self.TP_frontal = 0
        self.FP_frontal = 0
        self.TN_frontal = 0
        self.FN_frontal = 0
        
        self.TP_lateral = 0
        self.FP_lateral = 0
        self.TN_lateral = 0
        self.FN_lateral = 0
        
    '''====================================================================='''
    def increaseTPfrontal(self):
        self.TP_frontal += 1
    '''====================================================================='''
    def increaseFPfrontal(self):
        self.FP_frontal += 1
    '''====================================================================='''
    def increaseTNfrontal(self):
        self.TN_frontal += 1
    '''====================================================================='''
    def increaseFNfrontal(self):
        self.FN_frontal += 1
        
    '''====================================================================='''
    def increaseTPlateral(self):
        self.TP_lateral += 1
    '''====================================================================='''
    def increaseFPlateral(self):
        self.FP_lateral += 1
    '''====================================================================='''
    def increaseTNlateral(self):
        self.TN_lateral += 1
    '''====================================================================='''
    def increaseFNlateral(self):
        self.FN_lateral += 1
        
    '''====================================================================='''
    def getAccuracyFrontal(self):
        return (self.TP_frontal + self.TN_frontal) / (self.TP_frontal +
                                                      self.TN_frontal + 
                                                      self.FP_frontal + 
                                                      self.FN_frontal)
    
    '''====================================================================='''
    def getAccuracyLateral(self):
        return (self.TP_lateral + self.TN_lateral) / (self.TP_lateral +
                                                      self.TN_lateral + 
                                                      self.FP_lateral + 
                                                      self.FN_lateral)   
    
    '''====================================================================='''
    def getPrecisionFrontal(self):
        if self.TP_frontal == 0:
            return 0
        else:
            return self.TP_frontal / (self.TP_frontal + self.FP_frontal)  
    
    '''====================================================================='''
    def getPrecisionLateral(self):
        if self.TP_lateral == 0:
            return 0
        else:
            return self.TP_lateral / (self.TP_lateral + self.FP_lateral)  
    
    '''====================================================================='''
    def getRecallFrontal(self):
        if self.TP_frontal == 0:
            return 0
        else:
            return self.TP_frontal / (self.TP_frontal + self.FN_frontal)  
    
    '''====================================================================='''
    def getRecallLateral(self):
        if self.TP_lateral == 0:
            return 0
        else:
            return self.TP_lateral / (self.TP_lateral + self.FN_lateral)  
        
    '''====================================================================='''
    def getMccFrontal(self):
        denominator = math.sqrt((self.TP_frontal + self.FP_frontal) * 
                                (self.TP_frontal + self.FN_frontal) *
                                (self.TN_frontal + self.FP_frontal) *
                                (self.TN_frontal + self.FN_frontal))
        if denominator == 0.0:
            return 0
        else:
            return (self.TP_frontal * self.TN_frontal -
                    self.FP_frontal * self.FN_frontal) / denominator
    
    '''====================================================================='''
    def getMccLateral(self):
        denominator = math.sqrt((self.TP_lateral + self.FP_lateral) * 
                                (self.TP_lateral + self.FN_lateral) *
                                (self.TN_lateral + self.FP_lateral) *
                                (self.TN_lateral + self.FN_lateral))
        if denominator == 0.0:
            return 0
        else:
            return (self.TP_lateral * self.TN_lateral -
                    self.FP_lateral * self.FN_lateral) / denominator
    
    '''====================================================================='''
    def printAllStats(self):
        if self.compound:
            print('acc = {} ; prec = {} ; recall = {} ; mcc = {}'.format(self.getAccuracyFrontal(), self.getPrecisionFrontal(), self.getRecallFrontal(), self.getMccFrontal()))
        else:
            print('acc_front = {} ; prec_front = {} ; recall_front = {}; mcc_front = {}'.format(self.getAccuracyFrontal(), self.getPrecisionFrontal(), self.getRecallFrontal(), self.getMccFrontal()))
            print('acc_lat = {} ; prec_lat = {} ; recall_lat = {}; mcc_lat = {}'.format(self.getAccuracyLateral(), self.getPrecisionLateral(), self.getRecallLateral(), self.getMccLateral()))
