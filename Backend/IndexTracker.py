# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 10:43:23 2019

@author: mittmann
"""


class IndexTracker(object):
    def __init__(self, axes, image_data, image_name, positions):
        self.axes = axes
        self.image_data = image_data
        axes.set_title(image_name)
        self.index = 10
        self.positions = positions
        self.image = axes.imshow(self.image_data[:, :, self.index], cmap='gray', vmin=0, vmax=image_data.max())

        x_shape, y_shape, self.length = image_data.shape
        self.inverted = False
        self.update()

    def onscroll(self, event):
        if event.button == 'up':
            self.index = (self.index + 1) % self.length
        else:
            self.index = (self.index - 1) % self.length
        self.update()

    def update(self):
        if not self.inverted:
            self.inverted = True
            self.axes.invert_yaxis()

        self.image.set_data(self.image_data[:, :, self.index])
        self.axes.set_ylabel('index = %s' % self.index)
        self.image.axes.figure.canvas.draw()
        for point in self.positions:
            self.axes.plot(point[1], point[0], 'b+')
