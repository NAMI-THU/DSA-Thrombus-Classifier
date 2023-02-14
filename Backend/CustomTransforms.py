# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 17:01:39 2019

@author: mittmann
"""

import cv2
import torch
from albumentations import RandomRotate90

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

"""Convert ndarrays to Tensors"""


class ToTensor(object):

    def __call__(self, image_as_np_ndarray):
        # the axes have to be swapped as the convention is:
        # numpy ndarray image: height x width x channels
        # torch image: channels X height X width
        image = image_as_np_ndarray.transpose(2, 0, 1)
        return torch.from_numpy(image)


"""Convert Tensors to ndarrays"""


class ToNdarray(object):

    def __call__(self, image_as_tensor):
        # the axes have to be swapped as the convention is:
        # numpy ndarray image: height x width x channels
        # torch image: channels X height X width
        image = image_as_tensor.numpy()

        return image.transpose(1, 2, 0)


class Rotate90(RandomRotate90):
    def get_params(self):
        return {"factor": 1}
