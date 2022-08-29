#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 12:02:56 2020

@author: nami
"""

import timm
import torch.nn
import torch.nn.functional as F


class CnnLstmModel(torch.nn.Module):
    def __init__(self, hiddenSize, numLayers, outputSize, bidirectional, device):
        super(CnnLstmModel, self).__init__()
        # Hidden dimensions
        self.hidden_size = hiddenSize
        # Number of hidden layers
        self.num_layers = numLayers
        self.bidirectional = bidirectional
        self.device = device

        # Building the CNN + LSTM
        # Initialize CNN:

        # für Resnet18/34/50
        '''
        self.cnn = models.resnet18(pretrained=True) 
        self.inFeatures = self.cnn.fc.in_features * 49 
        self.cnn.avgpool = torch.nn.AdaptiveAvgPool2d((7, 7)) 
        self.cnn.fc = torch.nn.Identity()
        '''
        # für EfficientNet
        '''
        self.cnn = models.efficientnet_b1(pretrained=True)
        self.inFeatures = self.cnn.classifier[1].in_features * 16
        self.cnn.avgpool = torch.nn.AdaptiveAvgPool2d((4, 4))
        self.cnn.classifier = torch.nn.Identity()
        '''
        # für EfficientNet V2

        # pprint(timm.list_models(pretrained=True))
        self.cnn = timm.create_model('efficientnetv2_rw_s', pretrained=True)
        # self.cnn = timm.create_model('tf_efficientnetv2_m_in21k', pretrained=True)
        self.inFeatures = self.cnn.classifier.in_features * 16
        self.cnn.global_pool = torch.nn.AdaptiveAvgPool2d((4, 4))
        self.cnn.classifier = torch.nn.Identity()

        # für RegNet
        '''
        self.cnn = models.regnet_y_16gf(pretrained=True)
        self.inFeatures = self.cnn.fc.in_features * 9 
        self.cnn.avgpool = torch.nn.AdaptiveAvgPool2d((3, 3)) 
        self.cnn.fc = torch.nn.Identity()
        '''

        # print(self.cnn)
        ##print(self.inFeatures)

        # self.cnn.fc = torch.nn.Linear(self.cnn.fc.in_features, int(self.cnn.fc.in_features))

        self.layerNorm = torch.nn.LayerNorm(self.inFeatures)

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        ##self.lstm = torch.nn.LSTM(input_size=self.inFeatures, hidden_size=hiddenSize, num_layers=numLayers, bidirectional=self.bidirectional, dropout=0.5, batch_first=True)
        self.gru = torch.nn.GRU(input_size=self.inFeatures, hidden_size=hiddenSize, num_layers=numLayers,
                                bidirectional=self.bidirectional, dropout=0.5, batch_first=True)

        # Readout layer
        if bidirectional:
            # self.fc1 = torch.nn.Linear(2 * hiddenSize, 2 * hiddenSize)
            self.fc = torch.nn.Linear(2 * hiddenSize, outputSize)
        else:
            # self.fc1 = torch.nn.Linear(hiddenSize, hiddenSize)
            self.fc = torch.nn.Linear(hiddenSize, outputSize)

        # self.dropout1 = torch.nn.Dropout(p=0.3)
        # self.dropout2 = torch.nn.Dropout(p=0.5)
        # self.relu = torch.nn.ReLU()

    def forward(self, image_sequence):
        '''
        length = image_sequence.shape[1]
        #print(length)
        for index in range(0, length, 2):
            #print(index)
            if index + 2 < length:
                image = image_sequence[:, index:index + 3, :, :]
            elif index + 1 < length:
                image1 = image_sequence[:, index:index + 2, :, :]
                image2 = image_sequence[:, index + 1, :, :].view(-1, 1, 1024, 1024)
                image = torch.cat((image1, image2), dim=1)
            else:
                image1 = image_sequence[:, index, :, :].view(-1, 1, 1024, 1024)
                image2 = image_sequence[:, index, :, :].view(-1, 1, 1024, 1024)
                image3 = image_sequence[:, index, :, :].view(-1, 1, 1024, 1024)
                image = torch.cat((image1, image2, image3), dim=1)
            
            image = image.view(-1, 3, 1024, 1024)
            #print(image.shape)
            output_cnn = self.cnn(image.to(device=self.device, dtype=torch.float32))
            
            output_cnn = F.leaky_relu(self.layerNorm(output_cnn))
            print(output_cnn.shape)
            del image
            
            if index == 0:
                #h0 = torch.randn(self.num_layers * 2, output_cnn.size(0), self.hidden_size).requires_grad_()
                #c0 = torch.randn(self.num_layers * 2, output_cnn.size(0), self.hidden_size).requires_grad_()
                out, (hn) = self.gru(output_cnn.view(-1, 1, self.inFeatures))
            #elif index >= length - 40:
            #    out, (hn, cn) = self.lstm(output_cnn.view(-1, 1, self.inFeatures), (hn, cn))
            else:
                out, (hn) = self.gru(output_cnn.view(-1, 1, self.inFeatures), (hn.detach()))
                
        '''
        length = image_sequence.shape[1]
        if length % 3 == 2:
            image1 = image_sequence[:, -1, :, :].view(-1, 1, 512, 512)
            image_sequence = torch.cat((image_sequence, image1), dim=1)
            length += 1

        elif length % 3 == 1:
            image1 = image_sequence[:, -1, :, :].view(-1, 1, 512, 512)
            image_sequence = torch.cat((image_sequence, image1, image1), dim=1)
            length += 2

        image = image_sequence.view(int(length / 3), 3, 512, 512)
        output_cnn = self.cnn(image.to(device=self.device, dtype=torch.float32))
        # print(output_cnn.shape)
        output_cnn = torch.flatten(output_cnn, start_dim=1)
        output_cnn = F.leaky_relu(self.layerNorm(output_cnn))
        # print(output_cnn.shape)
        out, (hn) = self.gru(output_cnn.view(-1, int(length / 3), self.inFeatures))
        return self.fc(out[:, -1, :])
