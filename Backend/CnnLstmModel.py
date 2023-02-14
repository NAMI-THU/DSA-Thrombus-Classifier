#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 12:02:56 2020

@author: mittmann
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
        self.cnn = timm.create_model('efficientnetv2_rw_s', pretrained=True)
        self.inFeatures = self.cnn.classifier.in_features * 16
        self.cnn.global_pool = torch.nn.AdaptiveAvgPool2d((4, 4))
        self.cnn.classifier = torch.nn.Identity()
        self.layerNorm = torch.nn.LayerNorm(self.inFeatures)

        self.gru = torch.nn.GRU(input_size=self.inFeatures, hidden_size=hiddenSize, num_layers=numLayers,
                                bidirectional=self.bidirectional, dropout=0.5, batch_first=True)

        # Readout layer
        if bidirectional:
            self.fc = torch.nn.Linear(2 * hiddenSize, outputSize)
        else:
            self.fc = torch.nn.Linear(hiddenSize, outputSize)

    def forward(self, image_sequence):
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
        output_cnn = torch.flatten(output_cnn, start_dim=1)
        output_cnn = F.leaky_relu(self.layerNorm(output_cnn))
        out, (hn) = self.gru(output_cnn.view(-1, int(length / 3), self.inFeatures))
        return self.fc(out[:, -1, :])
