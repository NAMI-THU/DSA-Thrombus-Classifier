# -*- coding: utf-8 -*-

import numpy as np

class ImageUtils(object):

    @staticmethod
    def fillBlackBorderWithRandomNoise(image=np.ndarray((0,0,0)), mean=193):
        #create mask of outer black border:
        mask_image = np.ones((image.shape[0], image.shape[0]))
        mask = np.logical_not(np.logical_and(image[:,:,0], mask_image)) == True

        #create random filling of black border:
        #np.random.seed
        #noise = np.random.uniform(-0.07,0.07, image[mask].shape)
        noise_array = np.full(image[mask].shape, mean, np.uint8)
        #noise_array += noise
        #fill black border with random noise
        image[mask] = noise_array
        return image