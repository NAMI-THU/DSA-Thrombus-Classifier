# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 16:56:53 2019

@author: mittmann
"""

import os
import numpy as np
import nibabel
import matplotlib.pyplot as plt
import random
import torch
from torchvision import transforms
import cv2
import CustomTransforms
import albumentations as albu
from IndexTracker import IndexTracker

def fillBlackBorderWithRandomNoise(image, mean):
    #create mask of outer black border:
    size = image.shape[0]
    mask_image = np.ones((size, size))
    mask = np.logical_not(np.logical_and(image[:,:,0], mask_image)) == True

    #create random filling of black border:
    np.random.seed
    noise = np.random.uniform(-0.6,1.2, image[mask].shape)
    noise_array = np.full(image[mask].shape, mean, np.float64)
    noise_array += np.floor(60 * noise)
    #fill black border with random noise
    image[mask] = noise_array
    return image


#data_path = "C:\\Datasets\\Daten-Guenzburg\\DSA-aufbereitet-nifti"
data_path = "C:\\Daten-Guenzburg\\nifti"

files = os.listdir(data_path)

#for index, filename in enumerate(files):
filename = files[0]
dsa_image_path = os.path.join(data_path, filename)
dsa_image = nibabel.load(dsa_image_path)

image_data = dsa_image.get_fdata()
image_data = fillBlackBorderWithRandomNoise(image_data, 3050)

# image lateral view:
filename2 = files[1]
dsa_image_path2 = os.path.join(data_path, filename2)
dsa_image2 = nibabel.load(dsa_image_path2)

image_data2 = dsa_image2.get_fdata()
image_data2 = fillBlackBorderWithRandomNoise(image_data2, 3050)

#plot the DSA serie as scrollable image:
thrombus_positions = [(327, 606)]
thrombus_positions2 = [(637, 580)]
'''fig, ax = plt.subplots(1, 1)
tracker = IndexTracker(ax, image_data, filename, thrombus_positions)
fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
plt.show()
'''

#data = {'image': image_data, 'keypoints': thrombus_positions}

data = {'image': image_data,
        'keypoints': thrombus_positions,
        'imageLateralView': image_data2,
        'keypointsLateral': thrombus_positions2}

#albu.Rotate(limit=30, p=0, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=(3100,3100,3100,3100)),
#IAAEmboss kommt visuell sehr nahe an Bildartefakten dran, erhöht aber Wertebereich > 1.0

transform = albu.Compose([albu.ToFloat(max_value=4095.0),
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
                          albu.OneOf([
                            albu.MultiplicativeNoise(p=0.2,
                                                     multiplier=(0.9,1.1),
                                                     per_channel=True ,
                                                     elementwise=True),
                            albu.Downscale(p=0.2, scale_min=0.5, scale_max=0.5) #decreases image quality
                            ], p=0.5),
                          albu.IAAPiecewiseAffine(p=0.5, scale=(0.02, 0.02)) #Rechenintensiv, aber Deformationen werden gut abgebildet
                          ], p=1,
                          keypoint_params=albu.KeypointParams(format='yx'),
                          additional_targets={'imageLateralView': 'image',
                                              'keypointsLateral': 'keypoints'}                     
                          )
transformed_data = transform(**data)

transformed_image = transformed_data['image']
transformed_positions = transformed_data['keypoints']

transformed_image2 = transformed_data['imageLateralView']
transformed_positions2 = transformed_data['keypointsLateral']

'''
x = thrombus_positions[0][0]
y = thrombus_positions[0][1]
print(image_data[x,y,0])

if(transformed_positions):
    x_ = transformed_positions[0][0]
    y_ = transformed_positions[0][1]
    print(transformed_image[x_,y_,0])
'''

fig2, ax2 = plt.subplots(1, 1)
tracker2 = IndexTracker(ax2, transformed_image, filename, transformed_positions)
fig2.canvas.mpl_connect('scroll_event', tracker2.onscroll)
plt.show()

fig3, ax3 = plt.subplots(1, 1)
tracker3 = IndexTracker(ax3, transformed_image2, filename2, transformed_positions2)
fig3.canvas.mpl_connect('scroll_event', tracker3.onscroll)
plt.show()

#transforms.Compose([CustomTransforms.RandomFlipHorizontally(),
#                   transforms.ToPILImage()])

"""transformToTensor = CustomTransforms.ToTensor()
image_tensor = transformToTensor(image_data)

thrombus_positions = np.array(([327, 606],
                               [200, 200],
                               [800, 800]))

sample = {"dsa_series": image_tensor,
          "positions": thrombus_positions}
transformRandomFlip = CustomTransforms.RandomFlipHorizontally()
transformedSample = transformRandomFlip(sample)


#transformToNdarray = CustomTransforms.ToNdarray()
#image_ndarray = transformToNdarray(image_tensor)
"""
