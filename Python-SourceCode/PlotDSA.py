# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 16:56:53 2019

@author: mittmann
"""

import os
import numpy as np
import nibabel
import matplotlib.pyplot as plt
from IndexTracker import IndexTracker


#Pfad zu externen Festplatte (langsam):
#data_path = "F:\TE_GZ\DSA-aufbereitet-nifti"
#Netzwerkpfad (sehr langsam):
#data_path = "//hs-ulm.de/fs/archiv/IFI/Forschung/NAMI/Thrombomap/Datasets/TE_GZ/DSA-aufbereitet-nifti"

data_path = "C:\\Datasets\\Daten-Guenzburg\\DSA-aufbereitet-nifti"

files = os.listdir(data_path)

#for index, filename in enumerate(files):

filename = files[0]
dsa_image_path = os.path.join(data_path, filename)
dsa_image = nibabel.load(dsa_image_path)

image_data = dsa_image.get_fdata()

length_dsa_series = image_data.shape[2]
index_first_quartile = (int) (length_dsa_series / 4)
index_mean = 2 * index_first_quartile
index_third_quartile = 3 * index_first_quartile

thrombus_positions = np.array(([327, 606],
                               [200, 200],
                               [800, 800]))

#print(image_data[400, 7, 15])

''' Example 1: plotting one slice of DSA series as single big plot: '''
plt.figure()
plt.imshow(image_data[:,:,15], cmap='gray', vmin=0, vmax=4100)
plt.plot(606, 327, 'b+')

''' Example 2: plotting 4 slices of DSA series as 2x2 subplots: '''
figure, axes = plt.subplots(2, 2)
axes[0, 0].imshow(image_data[:,:,index_first_quartile], cmap='gray', vmin=0, vmax=4100)
axes[0, 0].plot(606, 327, 'b+')
axes[0, 0].set_title('Slice first quartile')

axes[0, 1].imshow(image_data[:,:,index_mean], cmap='gray', vmin=0, vmax=4100)
axes[0, 1].plot(606, 327, 'b+')
axes[0, 1].set_title('Slice mean')

axes[1, 0].imshow(image_data[:,:,index_third_quartile], cmap='gray', vmin=0, vmax=4100)
axes[1, 0].plot(606, 327, 'b+')
axes[1, 0].set_title('Slice third quartile')

axes[1, 1].imshow(image_data[:,:,-1], cmap='gray', vmin=0, vmax=4100)
axes[1, 1].plot(606, 327, 'b+')
axes[1, 1].set_title('Slice End')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axes.flat:
    ax.label_outer()


''' Example 3: plotting DSA serie as scrollable image: '''
fig, ax = plt.subplots(1, 1)
tracker = IndexTracker(ax, image_data, filename, thrombus_positions)
fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
plt.show()
