# -*- coding: utf-8 -*-
import os
import time

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


class Classificator:

    def __init__(self):
        torch.set_num_threads(8)
        self.models_loaded = False
        self.models_frontal = []
        self.models_lateral = []

    def load_models(self):
        MODEL_F = "models\\frontal"
        # MODEL_F = "models\\model_frontal.pt"
        # MODEL_L = "models\\model_lateral.pt"
        MODEL_L = "models\\lateral"

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Running on {device}")

        # Load Checkpoints:
        dir_list_f = os.listdir(MODEL_F)
        dir_list_l = os.listdir(MODEL_L)
        checkpoints_frontal = []
        checkpoints_lateral = []
        for m_f in dir_list_f:
            m_f = os.path.join(MODEL_F,m_f)
            checkpoints_frontal.append(torch.load(m_f, map_location=device))
        for m_l in dir_list_l:
            m_l = os.path.join(MODEL_L, m_l)
            checkpoints_lateral.append(torch.load(m_l, map_location=device))

        # Initialize model for frontal images:
        # model_frontal = models.resnet152()
        # model_frontal.conv1 = torch.nn.Conv2d(62, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # model_frontal.fc = torch.nn.Linear(model_frontal.fc.in_features, 1)
        # model_frontal = LSTMModel(1024*1024, 50, 2, 1, True)
        for m_f in checkpoints_frontal:
            model_frontal = CnnLstmModel(512, 3, 1, True, device)
            # model_frontal = torch.nn.DataParallel(model_frontal)
            model_frontal.load_state_dict(m_f['model_state_dict'])
            model_frontal.to(device)
            self.models_frontal.append(model_frontal)

        # Initialize model for lateral images:
        # model_lateral = models.resnet152()
        # model_lateral.conv1 = torch.nn.Conv2d(62, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # model_lateral.fc = torch.nn.Linear(model_lateral.fc.in_features, 1)
        # model_lateral = LSTMModel(1024*1024, 50, 2, 1, True)
        for m_l in checkpoints_lateral:
            model_lateral = CnnLstmModel(512, 3, 1, True, device)
            # model_lateral = torch.nn.DataParallel(model_lateral)
            model_lateral.load_state_dict(m_l['model_state_dict'])
            model_lateral.to(device)
            self.models_lateral.append(model_lateral)

        self.models_loaded = True

    def load_images(self, image_f, image_l):
        image_data = nibabel.load(image_f).get_fdata(caching='unchanged',
                                                     dtype=np.float32) * 0.062271062  # = 255/4095.0
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

    def do_classification(self, image_frontal, image_lateral):
        t0 = time.time()

        if not self.models_loaded:
            print("Models not prepared, loading them now...")
            self.load_models()

        IMAGE_L = image_lateral
        IMAGE_F = image_frontal

        # IMAGE_L = "images\\thrombYes\\263-01-aci-l-s.nii"
        # IMAGE_L = "images\\thrombNo\\095-03-aci-r-s.nii"
        # IMAGE_F = "images\\thrombYes\\263-01-aci-l-f.nii"
        # IMAGE_F = "images\\thrombNo\\095-03-aci-r-f.nii"

        t1 = time.time()
        data_prepared = self.load_images(IMAGE_F, IMAGE_L)

        images_frontal = torch.unsqueeze(data_prepared['image'], 0)
        images_lateral = torch.unsqueeze(data_prepared['imageOtherView'], 0)

        t2 = time.time()

        outputs_frontal = []
        outputs_lateral = []
        estimates_frontal = []
        estimates_lateral = []
        global_goal = len(self.models_frontal) + len(self.models_lateral)
        current_progress = 0
        for m_f in self.models_frontal:
            output_frontal = m_f(images_frontal)
            activation_f = torch.sigmoid(output_frontal).item()
            estimate_frontal = THROMBUS_NO if activation_f <= 0.5 else THROMBUS_YES
            outputs_frontal.append(activation_f)
            estimates_frontal.append(estimate_frontal)
            current_progress += 1
            print(f"Progress: {current_progress}/{global_goal} ({current_progress*100/global_goal}%)")
        for m_l in self.models_lateral:
            output_lateral = m_l(images_lateral)
            activation_l = torch.sigmoid(output_lateral).item()
            estimate_lateral = THROMBUS_NO if activation_l <= 0.5 else THROMBUS_YES
            outputs_lateral.append(activation_l)
            estimates_lateral.append(estimate_lateral)
            current_progress += 1
            print(f"Progress: {current_progress}/{global_goal} ({current_progress * 100 / global_goal}%)")

        t3 = time.time()
        del images_frontal
        del images_lateral


        # print(f"Estimate Frontal (has Thrombus?): {estimate_frontal == THROMBUS_YES} / Raw was {activation_f}")
        # print(f"Estimate Lateral (has Thrombus?): {estimate_lateral == THROMBUS_YES} / Raw was {activation_l}")

        print(
            f"Timings: \n Init model:{t1 - t0} ({(t1 - t0) * 100 / (t3 - t0)}%)\nLoad/Prepare Data: {t2 - t1} ({(t2 - t1) * 100 / (t3 - t0)}%)\nClassification: {t3 - t2} ({(t3 - t2) * 100 / (t3 - t0)}%)")

        return outputs_frontal, outputs_lateral, estimates_frontal, estimates_lateral


if __name__ == "__main__":
    c = Classificator()
    c.load_models()
    c.do_classification("images\\thrombYes\\263-01-aci-l-f.nii", "images\\thrombYes\\263-01-aci-l-s.nii")
