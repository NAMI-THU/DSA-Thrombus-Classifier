# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 16:56:53 2019

@author: mittmann
"""

import cv2
import numpy as np

import CustomTransforms
import albumentations as albu
import matplotlib.pyplot as plt
import torch
from IndexTracker import IndexTracker
import copy

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

'''Augments given data, which is stored in a dictionary'''


class DataAugmentation(object):
    MAX_SERIES_LENGTH = 62
    MAX_KEYPOINT_LENGTH = 5
    '''====================================================================='''

    def __init__(self, data=dict()):
        self.data = data
        self.transform = None
        self.tracker1 = None
        self.tracker2 = None
        self.tracker = None

    '''====================================================================='''

    def getTransform(self):
        return self.transform

    '''====================================================================='''

    def createTransformTraining(self):
        if not self.data:
            return

        if self.data['frontalAndLateralView']:
            self.transform = albu.Compose(
                [
                    albu.VerticalFlip(p=0.5),
                    albu.ShiftScaleRotate(p=1,
                                          shift_limit=0.15,
                                          scale_limit=(0, 0.1),
                                          rotate_limit=20,
                                          interpolation=cv2.INTER_LINEAR,
                                          border_mode=cv2.BORDER_CONSTANT,
                                          value=(193, 193, 193, 193)),  # previously: 0.757 for float32
                    CustomTransforms.Rotate90(p=1),  # always p=1
                    albu.Resize(512, 512, interpolation=cv2.INTER_LINEAR),
                    albu.RandomBrightnessContrast(p=0.2,
                                                  brightness_limit=0.2,
                                                  contrast_limit=0.2,
                                                  brightness_by_max=False),
                    albu.MedianBlur(p=0.5, blur_limit=3),
                    albu.OneOf(
                        [
                            albu.MultiplicativeNoise(p=0.2,
                                                     multiplier=(0.9, 1.1),
                                                     per_channel=True,
                                                     elementwise=True),
                            albu.Downscale(p=0.3, scale_min=0.5, scale_max=0.5)
                            # decreases image quality
                        ], p=0.5)
                ], p=1,
                keypoint_params=albu.KeypointParams(format='yx'),
                additional_targets={'imageOtherView': 'image',
                                    'keypointsOtherView': 'keypoints'},
            is_check_shapes = False
            )
        else:
            self.transform = albu.Compose(
                [
                    albu.VerticalFlip(p=0.5),
                    albu.ShiftScaleRotate(p=1,
                                          shift_limit=0.15,
                                          scale_limit=(0, 0.1),
                                          rotate_limit=20,
                                          interpolation=cv2.INTER_LINEAR,
                                          border_mode=cv2.BORDER_CONSTANT,
                                          value=(193, 193, 193, 193)),
                    CustomTransforms.Rotate90(p=1),
                    albu.Resize(512, 512, interpolation=cv2.INTER_LINEAR),
                    albu.RandomBrightnessContrast(p=0.2,
                                                  brightness_limit=0.2,
                                                  contrast_limit=0.2,
                                                  brightness_by_max=False),
                    albu.MedianBlur(p=0.5, blur_limit=3),
                    albu.OneOf(
                        [
                            albu.MultiplicativeNoise(p=0.2,
                                                     multiplier=(0.9, 1.1),
                                                     per_channel=True,
                                                     elementwise=True),
                            albu.Downscale(p=0.3, scale_min=0.5, scale_max=0.5)
                        ], p=0.5)
                ], p=1,
                keypoint_params=albu.KeypointParams(format='yx'),
                is_check_shapes=False
            )

    '''====================================================================='''

    def createTransformValidation(self):
        if not self.data:
            return

        if self.data['frontalAndLateralView']:
            self.transform = albu.Compose(
                [
                    CustomTransforms.Rotate90(p=1),
                    albu.Resize(512, 512, interpolation=cv2.INTER_LINEAR)
                ], p=1,
                keypoint_params=albu.KeypointParams(format='yx'),
                additional_targets={'imageOtherView': 'image',
                                    'keypointsOtherView': 'keypoints'},
                is_check_shapes=False
            )
        else:
            self.transform = albu.Compose(
                [
                    CustomTransforms.Rotate90(p=1),
                    albu.Resize(512, 512, interpolation=cv2.INTER_LINEAR)
                ], p=1,
                keypoint_params=albu.KeypointParams(format='yx'),
                is_check_shapes=False
            )

    '''====================================================================='''

    # @jit
    def applyTransform(self):
        if not self.transform:
            assert "Transform not yet created. Cannot apply transform."

        keypoints = copy.deepcopy(self.data['keypoints'])
        keypointsOtherView = copy.deepcopy(self.data['keypointsOtherView'])

        self.data = self.transform(**self.data)

        # Undo transformation of keypoints, if the keypoints indicated, that
        # there was no thrombus detected by the radiologist [(0,0)] or, that
        # the radiologist did not classify the image [(0,1)]:

        if (keypoints == [(0, 0)]) or (keypoints == [(0, 1)]):
            self.data['keypoints'] = keypoints
        if (keypointsOtherView == [(0, 0)]) or (keypointsOtherView == [(0, 1)]):
            self.data['keypointsOtherView'] = keypointsOtherView

        return

    '''====================================================================='''

    def getTransformedData(self):
        return self.data

    '''====================================================================='''

    # @jit(nopython=True)
    def adjustSequenceLengthBeforeTransform(self):
        if not self.data:
            raise RuntimeError("Data has to be set before \
                               performing adjusting")

        if self.data['image'] is not None and self.data['imageOtherView'] is not None:
            x1, y1, z1 = self.data['image'].shape
            x2, y2, z2 = self.data['imageOtherView'].shape

            if z1 != z2:
                if z1 > z2:
                    slices_to_add = z1 - z2
                    zeros = np.full((x2, y2, slices_to_add), 193, dtype=np.uint8)
                    self.data['imageOtherView'] = np.append(self.data['imageOtherView'], zeros, axis=2)
                else:
                    slices_to_add = z2 - z1
                    zeros = np.full((x1, y1, slices_to_add), 193, dtype=np.uint8)
                    self.data['image'] = np.append(self.data['image'], zeros, axis=2)
        return

    '''====================================================================='''

    # @jit(nopython=True)
    def zeroPaddingEqualLength(self):
        if not self.data:
            raise RuntimeError("Data has to be transformed first before \
                               performing zero padding")

        if self.data['image'] is not None:
            self.data['image'] = self.data['image'].astype(np.float32)

        if self.data['imageOtherView'] is not None:
            self.data['imageOtherView'] = self.data['imageOtherView'].astype(np.float32)

        if self.data['keypoints'] is not None:
            keypoints_to_add1 = self.MAX_KEYPOINT_LENGTH - len(self.data['keypoints'])
            for i in range(keypoints_to_add1):
                self.data['keypoints'].append((0, 0))

        if self.data['keypointsOtherView'] is not None:
            keypoints_to_add2 = self.MAX_KEYPOINT_LENGTH - len(self.data['keypointsOtherView'])
            for i in range(keypoints_to_add2):
                self.data['keypointsOtherView'].append((0, 0))

    '''====================================================================='''

    # @jit(nopython=True)
    def normalizeData(self):
        if self.data['image'] is not None:
            self.data['imageMean'] = np.mean(self.data['image'])
            self.data['imageStd'] = np.std(self.data['image'])
            self.data['image'] -= np.mean(self.data['image'], dtype=np.float32)
            self.data['image'] /= np.std(self.data['image'], dtype=np.float32)

        if self.data['imageOtherView'] is not None:
            self.data['imageOtherViewMean'] = np.mean(self.data['imageOtherView'])
            self.data['imageOtherViewStd'] = np.std(self.data['imageOtherView'])
            self.data['imageOtherView'] -= np.mean(self.data['imageOtherView'], dtype=np.float32)
            self.data['imageOtherView'] /= np.std(self.data['imageOtherView'], dtype=np.float32)

    '''====================================================================='''

    # @jit(nopython=True)
    def convertToTensor(self):
        if not self.data:
            raise RuntimeError("Data has to be transformed first before \
                               converting ToTensor")

        transformToTensor = CustomTransforms.ToTensor()

        self.data['image'] = transformToTensor(self.data['image'])
        self.data['keypoints'] = torch.tensor(self.data['keypoints'], dtype=torch.int32)
        if self.data['imageOtherView'] is not None:
            self.data['imageOtherView'] = transformToTensor(self.data['imageOtherView'])
        if self.data['keypointsOtherView'] is not None:
            self.data['keypointsOtherView'] = torch.tensor(self.data['keypointsOtherView'], dtype=torch.int32)

        return self.data

    '''====================================================================='''

    def showAugmentedImages(self):

        if self.data['frontalAndLateralView']:

            fig1, ax1 = plt.subplots(1, 1)
            self.tracker1 = IndexTracker(ax1, self.data['image'], 'Image1', self.data['keypoints'])
            fig1.canvas.mpl_connect('scroll_event', self.tracker1.onscroll)
            plt.show()

            fig2, ax2 = plt.subplots(1, 1)
            self.tracker2 = IndexTracker(ax2, self.data['imageOtherView'], 'ImageOtherView',
                                         self.data['keypointsOtherView'])
            fig2.canvas.mpl_connect('scroll_event', self.tracker2.onscroll)
            plt.show()

        else:
            fig, ax = plt.subplots(1, 1)
            self.tracker = IndexTracker(ax, self.data['image'], 'Image', self.data['keypoints'])
            fig.canvas.mpl_connect('scroll_event', self.tracker.onscroll)
            plt.show()

    def getImageData(self):
        seq_length = self.data['image'].shape[2]
        return np.copy(self.data['image'][:, :, int(seq_length / 2)]), np.copy(
            self.data['imageOtherView'][:, :, int(seq_length / 2)])
