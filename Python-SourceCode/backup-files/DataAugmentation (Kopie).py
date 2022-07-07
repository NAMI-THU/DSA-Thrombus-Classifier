# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 16:56:53 2019

@author: mittmann
"""


import cv2
import numpy as np
import CustomTransforms
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import albumentations as albu
import matplotlib.pyplot as plt
import torch.tensor
from IndexTracker import IndexTracker


'''Augments given data, which is stored in a dictionary'''
class DataAugmentation(object):
    MAX_SERIES_LENGTH = 62
    MAX_KEYPOINT_LENGTH = 5
    '''====================================================================='''
    def __init__(self, data=dict()):
        self.data = data
        self.transformed_data = dict()
        self.transform = None
        self.tracker1 = None
        self.tracker2 = None
        self.tracker = None

    '''====================================================================='''
    def getTransform(self):
        return self.transform

    '''====================================================================='''
    #albu.Rotate(limit=30, p=0, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=(3100,3100,3100,3100)),
    #IAAEmboss kommt visuell sehr nahe an Bildartefakten dran, erhöht aber Wertebereich > 1.0
    def createTransform(self):
        if not self.data:
            return

        if self.data['frontalAndLateralView'] == True:
            self.transform = albu.Compose(
                             [
                                 albu.ToFloat(max_value=4095.0),
                                 albu.VerticalFlip(p=0.5),
                                 albu.ShiftScaleRotate(p=0.5,
                                                       shift_limit=0.15,
                                                       scale_limit=(0,0.1),
                                                       rotate_limit=20,
                                                       interpolation=cv2.INTER_LINEAR,
                                                       border_mode=cv2.BORDER_CONSTANT,
                                                       value=(0.757,0.757,0.757,0.757)),
                                 CustomTransforms.Rotate90(p=1), #immer p=1
                                 albu.RandomBrightnessContrast(p=0.5,
                                                               brightness_limit=0.2,
                                                               contrast_limit=0.2,
                                                               brightness_by_max=False),
                                 albu.MedianBlur(p=0.2, blur_limit=3),
                                 albu.OneOf(
                                 [
                                     albu.MultiplicativeNoise(p=0.2,
                                                              multiplier=(0.9,1.1),
                                                              per_channel=True ,
                                                              elementwise=True),
                                     albu.Downscale(p=0.2, scale_min=0.5, scale_max=0.5) #decreases image quality
                                 ], p=0.5),
                                 albu.IAAPiecewiseAffine(p=0, scale=(0.02, 0.02)) #bisher: p=0.5, Rechenintensiv, aber Deformationen werden gut abgebildet
                             ], p=1,
                             keypoint_params=albu.KeypointParams(format='yx'),
                             additional_targets={'imageOtherView': 'image',
                                                 'keypointsOtherView': 'keypoints'}
                             )
        else:
            self.transform = albu.Compose(
                             [
                                 albu.ToFloat(max_value=4095.0),
                                 albu.VerticalFlip(p=0.5),
                                 albu.ShiftScaleRotate(p=0.5,
                                                       shift_limit=0.15,
                                                       scale_limit=(0,0.1),
                                                       rotate_limit=20,
                                                       interpolation=cv2.INTER_LINEAR,
                                                       border_mode=cv2.BORDER_CONSTANT,
                                                       value=(0.757,0.757,0.757,0.757)),
                                 CustomTransforms.Rotate90(p=1), #immer p=1
                                 albu.RandomBrightnessContrast(p=0.5,
                                                               brightness_limit=0.2,
                                                               contrast_limit=0.2,
                                                               brightness_by_max=False),
                                 albu.MedianBlur(p=0.2, blur_limit=3),
                                 albu.OneOf(
                                 [
                                     albu.MultiplicativeNoise(p=0.2,
                                                              multiplier=(0.9,1.1),
                                                              per_channel=True ,
                                                              elementwise=True),
                                     albu.Downscale(p=0.2, scale_min=0.5, scale_max=0.5) #decreases image quality
                                 ], p=0.5),
                                 albu.IAAPiecewiseAffine(p=0, scale=(0.02, 0.02)) #bisher: p=0.5, Rechenintensiv, aber Deformationen werden gut abgebildet
                             ], p=1,
                             keypoint_params=albu.KeypointParams(format='yx')
                             )

    '''====================================================================='''
    def applyTransform(self):
        if not self.transform:
            assert("Transform not yet created. Cannot apply transform.")
        
        keypoints = self.data['keypoints'].deepcopy()
        keypointsOtherView = self.data['keypointsOtherView'].deepcopy()
        
        self.data = self.transform(**self.data)

        # Undo transformation of keypoints, if the keypoints indicated, that
        # there was no thrombus detected by the radiologist [(0,0)] or, that
        # the radiologist did not classify the image [(0,1)]:

        if (keypoints == [(0, 0)]) or (keypoints == [(0, 1)]):
            self.data['keypoints'] = keypoints
            #print("Undo keypoint transformation as it is special keypoint")
        if (keypointsOtherView == [(0, 0)]) or (keypointsOtherView == [(0, 1)]):
            self.data['keypointsOtherView'] = keypointsOtherView
            #print("Undo keypoint transformation as it is special keypoint")

        return
    '''====================================================================='''
    def getTransformedData(self):
        return self.transformed_data

    '''====================================================================='''
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
                    zeros = np.zeros((x2, y2, slices_to_add))
                    self.data['imageOtherView'] = np.append(self.data['imageOtherView'], zeros, axis=2)
                else:
                    slices_to_add = z2 - z1
                    zeros = np.zeros((x1, y1, slices_to_add))
                    self.data['image'] = np.append(self.data['image'], zeros, axis=2)
        return
    
    '''====================================================================='''
    def zeroPaddingEqualLength(self):
        if not self.transformed_data:
            raise RuntimeError("Data has to be transformed first before \
                               performing zero padding")
        if self.transformed_data['image'] is not None:
            x1, y1, z1 = self.transformed_data['image'].shape
            slices_to_add1 = self.MAX_SERIES_LENGTH - z1
            zeros1 = np.zeros((x1, y1, slices_to_add1))

            #print(self.transformed_data['image'].shape)
            self.transformed_data['image'] = np.append(self.transformed_data['image'], zeros1, axis=2)
            #print(self.transformed_data['image'].shape)

        if self.transformed_data['imageOtherView'] is not None:
            x2, y2, z2 = self.transformed_data['imageOtherView'].shape
            slices_to_add2 = self.MAX_SERIES_LENGTH - z2
            zeros2 = np.zeros((x2, y2, slices_to_add2))

            #print(self.transformed_data['imageOtherView'].shape)
            self.transformed_data['imageOtherView'] = np.append(self.transformed_data['imageOtherView'], zeros2, axis=2)
            #print(self.transformed_data['imageOtherView'].shape)

        if self.transformed_data['keypoints'] is not None:
            keypoints_to_add1 = self.MAX_KEYPOINT_LENGTH - len(self.transformed_data['keypoints'])
            for i in range(keypoints_to_add1):
                self.transformed_data['keypoints'].append((0,0))
            #print(len(self.transformed_data['keypoints']))

        if self.transformed_data['keypointsOtherView'] is not None:
            keypoints_to_add2 = self.MAX_KEYPOINT_LENGTH - len(self.transformed_data['keypointsOtherView'])
            for i in range(keypoints_to_add2):
                self.transformed_data['keypointsOtherView'].append((0,0))
            #print(len(self.transformed_data['keypointsOtherView']))

    '''====================================================================='''
    def convertToTensor(self):
        if not self.transformed_data:
            raise RuntimeError("Data has to be transformed first before \
                               converting ToTensor")

        transformToTensor = CustomTransforms.ToTensor()

        self.transformed_data['image'] = transformToTensor(self.transformed_data['image'])
        self.transformed_data['keypoints'] = torch.tensor(self.transformed_data['keypoints'], dtype=torch.int32)
        if self.transformed_data['imageOtherView'] is not None:
            self.transformed_data['imageOtherView'] = transformToTensor(self.transformed_data['imageOtherView'])
        if self.transformed_data['keypointsOtherView'] is not None:
            self.transformed_data['keypointsOtherView'] = torch.tensor(self.transformed_data['keypointsOtherView'], dtype=torch.int32)

        return self.transformed_data

    '''====================================================================='''
    def showAugmentedImages(self):

        if self.data['frontalAndLateralView'] == True:
            
            fig1, ax1 = plt.subplots(1, 1)
            self.tracker1 = IndexTracker(ax1, self.transformed_data['image'], 'Image1', self.transformed_data['keypoints'])
            fig1.canvas.mpl_connect('scroll_event', self.tracker1.onscroll)
            plt.show()

            fig2, ax2 = plt.subplots(1, 1)
            self.tracker2 = IndexTracker(ax2, self.transformed_data['imageOtherView'], 'ImageOtherView', self.transformed_data['keypointsOtherView'])
            fig2.canvas.mpl_connect('scroll_event', self.tracker2.onscroll)
            plt.show()

        else:
            fig, ax = plt.subplots(1, 1)
            self.tracker = IndexTracker(ax, self.transformed_data['image'], 'Image', self.transformed_data['keypoints'])
            fig.canvas.mpl_connect('scroll_event', self.tracker.onscroll)
            plt.show()