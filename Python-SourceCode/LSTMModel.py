#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 10:26:15 2020

@author: nami
"""

import torch.nn

class LSTMModel(torch.nn.Module):
    def __init__(self, inputSize, hiddenSize, numLayers, outputSize, bidirectional):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_size = hiddenSize

        # Number of hidden layers
        self.num_layers = numLayers
        
        self.bidirectional = bidirectional

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = torch.nn.LSTM(input_size=inputSize, hidden_size=hiddenSize, num_layers=numLayers, bidirectional=bidirectional, batch_first=True)

        # Readout layer
        if bidirectional:
            #self.fc1 = torch.nn.Linear(2 * hiddenSize, 2 * hiddenSize)
            self.fc = torch.nn.Linear(2 * hiddenSize, outputSize)
        else:
            #self.fc1 = torch.nn.Linear(hiddenSize, hiddenSize)
            self.fc = torch.nn.Linear(hiddenSize, outputSize)

    def forward(self, x):
        # Initialize hidden state with zeros
        #h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        #h0.cuda()
        
        # Initialize cell state
        #c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        #c0.cuda()

        out, (hn, cn) = self.lstm(x)#, (h0.cuda(), c0.cuda()))
       
        out = self.fc(out[:, -1, :]) 
        # out.size() --> 100, 10
        return out