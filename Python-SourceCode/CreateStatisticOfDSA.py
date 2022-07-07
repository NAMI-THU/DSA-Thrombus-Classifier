# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 16:56:53 2019

@author: mittmann
"""

import os
import numpy as np
import nibabel

#Pfad zu externen Festplatte (langsam):
#data_path = "F:\TE_GZ\DSA-aufbereitet-nifti"
#Netzwerkpfad (sehr langsam):
#data_path = "//hs-ulm.de/fs/archiv/IFI/Forschung/NAMI/Thrombomap/Datasets/TE_GZ/DSA-aufbereitet-nifti"

data_path = "C:\\Datasets\\Daten-Guenzburg\\DSA-aufbereitet-nifti"

files = os.listdir(data_path)


shape_x = np.zeros(len(files), dtype=np.uint16)
shape_y = np.zeros(len(files), dtype=np.uint16)
shape_z = np.zeros(len(files), dtype=np.uint16)

data_type_count_uint16 = 0
data_type_others = 0

for index, filename in enumerate(files):
    dsa_image_path = os.path.join(data_path, filename)
    dsa_image = nibabel.load(dsa_image_path)

    shape_x[index], shape_y[index], shape_z[index] = dsa_image.shape

    if dsa_image.get_data_dtype() == np.dtype(np.uint16):
        data_type_count_uint16 += 1
    else:
        data_type_others += 1

    """image_data = dsa_image.get_fdata()"""
    #print(image_data[34, 32, 0])
    """print("Min-value:")
    print(image_data.min())
    print("Max-value:")
    print(image_data.max())"""

print("Shape_x Min Max =")
print(shape_x.min())
print(shape_x.max())
print("Shape_y Min Max =")
print(shape_y.min())
print(shape_y.max())
print("Shape_z Min Max =")
print(shape_z.min())
print(shape_z.max())
print("Shape_z mean =")
print(shape_z.mean())
print("3-Prozent 97-Prozent Quantil =")
print(np.percentile(shape_z, 3))
print(np.percentile(shape_z, 97))


