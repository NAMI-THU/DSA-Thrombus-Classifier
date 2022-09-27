# -*- coding: utf-8 -*-
import math
import os
import time

import nibabel
import numpy as np
import torch.cuda
import torch.nn
import torch.optim

from CnnLstmModel import CnnLstmModel
from DataAugmentation import DataAugmentation
from ImageUtils import ImageUtils

THROMBUS_NO = 0.214
THROMBUS_YES = 0.786

MINIMUM_FREE_GPU_VRAM_GB = 4


class Classificator:

    def __init__(self):
        torch.set_num_threads(8)
        self.models_loaded = {'f': "", 'l': ""}
        self.models_frontal = {}
        self.models_lateral = {}
        self.preparedImages = {}

        if torch.cuda.is_available():
            print("## CUDA is available. ##")
        else:
            print("## CUDA is not available on this machine. ##")

    def load_models(self, folder="models"):
        model_f = os.path.join(folder, "frontal")
        model_l = os.path.join(folder, "lateral")

        del self.models_lateral
        del self.models_frontal
        self.models_lateral = {}
        self.models_frontal = {}

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for d in range(torch.cuda.device_count()):
            device = torch.device(f"cuda:{d}")
            (free, total) = torch.cuda.mem_get_info(device)
            gb = free * 9.31 * math.pow(10, -10)
            print(f"Device {device} has {gb} free RAM.")
            if gb < MINIMUM_FREE_GPU_VRAM_GB:
                print(f"This is not enough, we need at least {MINIMUM_FREE_GPU_VRAM_GB} GB.")
                device = torch.device("cpu")
            else:
                device = torch.device(f"cuda:{d}")

        print(f"Running on {device}")

        # Load Checkpoints:
        dir_list_f = os.listdir(model_f)
        dir_list_l = os.listdir(model_l)

        # Initialize model for frontal images:
        # model_frontal = models.resnet152()
        # model_frontal.conv1 = torch.nn.Conv2d(62, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # model_frontal.fc = torch.nn.Linear(model_frontal.fc.in_features, 1)
        # model_frontal = LSTMModel(1024*1024, 50, 2, 1, True)
        for m_f_orig in dir_list_f:
            m_f = os.path.join(model_f, m_f_orig)
            model_frontal = CnnLstmModel(512, 3, 1, True, device)
            checkpoint = torch.load(m_f, map_location=device)
            # model_frontal = torch.nn.DataParallel(model_frontal)
            model_frontal.load_state_dict(checkpoint['model_state_dict'])
            model_frontal.to(device)
            self.models_frontal[m_f_orig] = model_frontal

        # Initialize model for lateral images:
        # model_lateral = models.resnet152()
        # model_lateral.conv1 = torch.nn.Conv2d(62, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # model_lateral.fc = torch.nn.Linear(model_lateral.fc.in_features, 1)
        # model_lateral = LSTMModel(1024*1024, 50, 2, 1, True)
        for m_l_orig in dir_list_l:
            m_l = os.path.join(model_l, m_l_orig)
            model_lateral = CnnLstmModel(512, 3, 1, True, device)
            checkpoint = torch.load(m_l, map_location=device)
            # model_lateral = torch.nn.DataParallel(model_lateral)
            model_lateral.load_state_dict(checkpoint['model_state_dict'])
            model_lateral.to(device)
            self.models_lateral[m_l_orig] = model_lateral

        self.models_loaded = {'f': model_f, 'l': model_l}

    def check_models_already_loaded(self, folder):
        new_f = os.path.join(folder, "frontal")
        new_l = os.path.join(folder, "lateral")
        return self.models_loaded['f'] == new_f and self.models_loaded['l'] == new_l

    def load_images(self, image_f, image_l, return_normalized=False):
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

        if not return_normalized:
            img1, img2 = augmentation.getImageData()

        augmentation.zeroPaddingEqualLength()
        augmentation.normalizeData()

        if return_normalized:
            img1, img2 = augmentation.getImageData()

        return augmentation.convertToTensor(), {'img1': img1, 'img2': img2}

    def prepare_images(self, image_frontal, image_lateral, normalized):
        data_prepared, imagedict = self.load_images(image_frontal, image_lateral, normalized)
        self.preparedImages[hash(image_frontal + image_lateral)] = data_prepared
        return imagedict

    def do_classification(self, image_f, image_l, mf="", ml=""):
        t0 = time.time()

        if not self.models_loaded:
            print("Models not prepared, loading them now...")
            self.load_models()

        h = hash(image_f + image_l)

        if not self.preparedImages[h]:
            raise Exception("Images have not been loaded yet")

        t1 = time.time()

        images_frontal = torch.unsqueeze(self.preparedImages[h]['image'], 0)
        images_lateral = torch.unsqueeze(self.preparedImages[h]['imageOtherView'], 0)

        outputs_frontal = []
        outputs_lateral = []
        estimates_frontal = []
        estimates_lateral = []
        global_goal = len(self.models_frontal) + len(self.models_lateral)
        current_progress = 0
        if mf == "":
            for m_f in self.models_frontal:
                output_frontal = m_f(images_frontal)
                activation_f = torch.sigmoid(output_frontal).item()
                del output_frontal
                torch.cuda.empty_cache()
                estimate_frontal = THROMBUS_NO if activation_f <= 0.5 else THROMBUS_YES
                outputs_frontal.append(activation_f)
                estimates_frontal.append(estimate_frontal)
                current_progress += 1
                print(f"Progress: {current_progress}/{global_goal} ({current_progress * 100 / global_goal}%)")
        else:
            output_frontal = self.models_frontal[mf](images_frontal)
            activation_f = torch.sigmoid(output_frontal).item()
            del output_frontal
            torch.cuda.empty_cache()
            estimate_frontal = THROMBUS_NO if activation_f <= 0.5 else THROMBUS_YES
            outputs_frontal.append(activation_f)
            estimates_frontal.append(estimate_frontal)

        if ml == "":
            for m_l in self.models_lateral:
                output_lateral = m_l(images_lateral)
                activation_l = torch.sigmoid(output_lateral).item()
                del output_lateral
                torch.cuda.empty_cache()
                estimate_lateral = THROMBUS_NO if activation_l <= 0.5 else THROMBUS_YES
                outputs_lateral.append(activation_l)
                estimates_lateral.append(estimate_lateral)
                current_progress += 1
                print(f"Progress: {current_progress}/{global_goal} ({current_progress * 100 / global_goal}%)")
        else:
            output_lateral = self.models_lateral[ml](images_lateral)
            activation_l = torch.sigmoid(output_lateral).item()
            del output_lateral
            torch.cuda.empty_cache()
            estimate_lateral = THROMBUS_NO if activation_l <= 0.5 else THROMBUS_YES
            outputs_lateral.append(activation_l)
            estimates_lateral.append(estimate_lateral)
        t2 = time.time()
        del images_frontal
        del images_lateral

        # TODO: Possible memory leak: remove images (maybe only when new ones are loaded?)

        print(
            f"=== Timings: === \n Init model:{t1 - t0} ({(t1 - t0) * 100 / (t2 - t0)}%)\nClassification: {t2 - t1} ({(t2 - t1) * 100 / (t2 - t0)}%)")

        return outputs_frontal, outputs_lateral, estimates_frontal, estimates_lateral


if __name__ == "__main__":
    c = Classificator()
    c.load_models()
    c.do_classification("images\\thrombYes\\263-01-aci-l-f.nii", "images\\thrombYes\\263-01-aci-l-s.nii")
