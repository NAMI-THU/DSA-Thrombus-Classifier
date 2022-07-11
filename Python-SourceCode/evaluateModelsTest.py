# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 16:56:53 2019

@author: mittmann
"""
# import cProfile
# import pstats
# from pstats import SortKey
import os

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
import nibabel
from ImageUtils import ImageUtils
from DataAugmentation import DataAugmentation

THROMBUS_NO = 0.214
THROMBUS_YES = 0.786


def load_images(image_f, image_l):
    image_data = nibabel.load(image_f).get_fdata(caching='unchanged', dtype=np.float32) * 0.062271062  # = 255/4095.0
    image_data = image_data.astype(np.uint8)
    image_data = ImageUtils.fillBlackBorderWithRandomNoise(image_data, 193)

    image_data2 = nibabel.load(image_l).get_fdata(
        caching='unchanged', dtype=np.float32) * 0.062271062
    image_data2 = image_data2.astype(np.uint8)
    image_data2 = ImageUtils.fillBlackBorderWithRandomNoise(image_data2, 193)

    data = {'image': image_data,
            'keypoints': [],
            'imageOtherView': image_data2,
            'keypointsOtherView': [],
            'frontalAndLateralView': True,
            'imageMean': 0,
            'imageOtherViewMean': 0,
            'imageStd': 1.0,
            'imageOtherViewStd': 1.0}

    augmentation = DataAugmentation(data=data)
    augmentation.adjustSequenceLengthBeforeTransform()
    augmentation.createTransformValidation()

    augmentation.applyTransform()

    augmentation.zeroPaddingEqualLength()
    augmentation.normalizeData()

    return augmentation.convertToTensor()


if __name__ == "__main__":
    torch.set_num_threads(8)

    MODEL_F = "models\\model_frontal.pt"
    MODEL_L = "models\\model_lateral.pt"
    IMAGE_L = "images\\263-01-aci-l-s.nii"
    IMAGE_F = "images\\263-01-aci-l-f.nii"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    # Load Checkpoints:
    checkpoint_frontal = torch.load(MODEL_F, map_location=device)
    checkpoint_lateral = torch.load(MODEL_L, map_location=device)

    # Initialize model for frontal images:
    # model_frontal = models.resnet152()
    # model_frontal.conv1 = torch.nn.Conv2d(62, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # model_frontal.fc = torch.nn.Linear(model_frontal.fc.in_features, 1)
    # model_frontal = LSTMModel(1024*1024, 50, 2, 1, True)
    model_frontal = CnnLstmModel(512, 3, 1, True, device)
    # model_frontal = torch.nn.DataParallel(model_frontal)
    model_frontal.load_state_dict(checkpoint_frontal['model_state_dict'])
    model_frontal.to(device)

    # Initialize model for lateral images:
    # model_lateral = models.resnet152()
    # model_lateral.conv1 = torch.nn.Conv2d(62, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # model_lateral.fc = torch.nn.Linear(model_lateral.fc.in_features, 1)
    # model_lateral = LSTMModel(1024*1024, 50, 2, 1, True)
    model_lateral = CnnLstmModel(512, 3, 1, True, device)
    # model_lateral = torch.nn.DataParallel(model_lateral)
    model_lateral.load_state_dict(checkpoint_lateral['model_state_dict'])
    model_lateral.to(device)

    data_prepared = load_images(IMAGE_F, IMAGE_L)

    images_frontal = torch.unsqueeze(data_prepared['image'], 0)
    images_lateral = torch.unsqueeze(data_prepared['imageOtherView'], 0)

    output_frontal = model_frontal(images_frontal)
    output_lateral = model_lateral(images_lateral)

    del images_frontal
    del images_lateral

    estimate_frontal = THROMBUS_NO if torch.sigmoid(output_frontal).item() <= 0.5 else THROMBUS_YES
    estimate_lateral = THROMBUS_NO if torch.sigmoid(output_lateral).item() <= 0.5 else THROMBUS_YES

    print(f"Estimate Frontal (has Thrombus?): {estimate_frontal == THROMBUS_YES}")
    print(f"Estimate Lateral (has Thrombus?): {estimate_lateral == THROMBUS_YES}")
