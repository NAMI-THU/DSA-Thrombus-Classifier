# -*- coding: utf-8 -*-

import numpy as np


class ImageUtils(object):

    @staticmethod
    def fillBlackBorderWithRandomNoise(image=np.ndarray((0, 0, 0)), mean=193):
        # create mask of outer black border:
        mask_image = np.ones((image.shape[0], image.shape[0]))
        mask = np.logical_not(np.logical_and(image[:, :, 0], mask_image))

        # create random filling of black border:
        noise_array = np.full(image[mask].shape, mean, np.uint8)
        # fill black border with random noise
        image[mask] = noise_array
        return image
