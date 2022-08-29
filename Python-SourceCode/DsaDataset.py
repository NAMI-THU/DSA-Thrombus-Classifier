# -*- coding: utf-8 -*-
"""
@author: mittmann
"""

import os

import nibabel
import numpy as np
import pandas as pd
import torch.utils.data

from DataAugmentation import DataAugmentation
from ImageUtils import ImageUtils


class DsaDataset(torch.utils.data.Dataset):
    def __init__(self, data_path="", csv_path="", data_set_name="", training=True):
        self.dataPath = data_path
        self.csvPath = csv_path
        self.data_set_name = data_set_name
        self.csvData = None  # is a multiindex dataframe containing all csv data
        self.imageNames = list()
        self.datasetDict = list()  # list with dictionary entries containing all information of dataset
        self.currentIndex = -1  # Standard: -1, für Entwicklungszwecke > 0
        self.files = os.listdir(self.dataPath)
        self.frontalAndLateralViewLastIndex = False
        self.training = training
        self.countThrombusfree = 0
        self.countThrombusyes = 0

    '''====================================================================='''

    def createDatasetDict(self):
        self.datasetDict.clear()
        if len(self.imageNames) == 0:
            raise RuntimeError("Cannot get next entry. \
                                The csv-file has to be read first.")
            return
        index = -1
        frontalAndLateralViewLastIndex = False

        while index < len(self.imageNames):
            if frontalAndLateralViewLastIndex == True:
                index += 2
            else:
                index += 1

            # Stellt sicher, dass keine out-of-index-error auftritt:
            if index >= len(self.imageNames):
                break

            # for index, filename in enumerate(files):
            filename = self.imageNames[index] + ".nii"
            positions = self.extractThrombusPositions(self.imageNames[index])
            # print(positions)

            # image other view:
            # check, if other view exists:
            if (index + 1) >= len(self.imageNames):
                frontalAndLateralViewLastIndex = False
                entry = {'filename': filename,
                         'keypoints': positions,
                         'filenameOtherView': None,
                         'keypointsOtherView': None,
                         'frontalAndLateralView': False}
                self.datasetDict.append(entry)
                break

            filename2 = self.imageNames[index + 1] + ".nii"

            # print(filename + " | " + filename2)
            # print(filename2[:7] in filename)

            if (filename2[:7] in filename) == False:
                # means: second image does not belong to the actual image series,
                # as the files do not have the same filename starting with.
                frontalAndLateralViewLastIndex = False
                entry = {'filename': filename,
                         'keypoints': positions,
                         'filenameOtherView': None,
                         'keypointsOtherView': None,
                         'frontalAndLateralView': False}
                self.datasetDict.append(entry)
                continue

            # means: second image belongs to actual image series. So, load it and
            #        append it to datasetDict.
            frontalAndLateralViewLastIndex = True
            positions2 = self.extractThrombusPositions(self.imageNames[index + 1])
            # print(positions2)

            entry = {'filename': filename,
                     'keypoints': positions,
                     'filenameOtherView': filename2,
                     'keypointsOtherView': positions2,
                     'frontalAndLateralView': True}
            if positions == [(0, 0)] and positions2 == [(0, 0)]:
                self.countThrombusfree += 1
            else:
                self.countThrombusyes += 1

            # print(index)
            # print(filename)
            self.datasetDict.append(entry)

        return

    '''====================================================================='''

    def __getitem__(self, index):
        entry = self.datasetDict[index]

        image_data = nibabel.load(os.path.join(self.dataPath, entry['filename'])).get_fdata(caching='unchanged',
                                                                                            dtype=np.float32) * 0.062271062  # = 255/4095.0
        image_data = image_data.astype(np.uint8)
        # print(image_data[400,600,10])
        image_data = ImageUtils.fillBlackBorderWithRandomNoise(image_data, 193)

        if entry['frontalAndLateralView'] == False:
            data = {'image': image_data,
                    'keypoints': entry['keypoints'],
                    'imageOtherView': None,
                    'keypointsOtherView': None,
                    'frontalAndLateralView': False,
                    'imageMean': 0,
                    'imageOtherViewMean': 0,
                    'imageStd': 1.0,
                    'imageOtherViewStd': 1.0}
        else:
            image_data2 = nibabel.load(os.path.join(self.dataPath, entry['filenameOtherView'])).get_fdata(
                caching='unchanged', dtype=np.float32) * 0.062271062
            image_data2 = image_data2.astype(np.uint8)
            # print(image_data2[400,600,10])
            image_data2 = ImageUtils.fillBlackBorderWithRandomNoise(image_data2, 193)

            data = {'image': image_data,
                    'keypoints': entry['keypoints'],
                    'imageOtherView': image_data2,
                    'keypointsOtherView': entry['keypointsOtherView'],
                    'frontalAndLateralView': True,
                    'imageMean': 0,
                    'imageOtherViewMean': 0,
                    'imageStd': 1.0,
                    'imageOtherViewStd': 1.0}

        augmentation = DataAugmentation(data=data)
        augmentation.adjustSequenceLengthBeforeTransform()

        if self.training:
            augmentation.createTransformTraining()
        else:
            augmentation.createTransformValidation()

        # augmentation.createTransformValidation()

        augmentation.applyTransform()

        augmentation.zeroPaddingEqualLength()
        augmentation.normalizeData()

        return augmentation.convertToTensor()

    '''====================================================================='''

    def __len__(self):
        return len(self.datasetDict)

    '''====================================================================='''

    def getCurrentIndex(self):
        return self.currentIndex

    '''====================================================================='''

    def getFrontalAndLateralViewLastIndex(self):
        return self.frontalAndLateralViewLastIndex

    '''====================================================================='''

    def loadCsvData(self):
        path = os.path.join(self.csvPath, self.data_set_name)

        data = pd.read_csv(path,
                           sep=';',
                           dtype={'ImageName': str,
                                  'PointNo': np.int64,
                                  'x': np.int64,
                                  'y': np.int64},
                           usecols=['ImageName', 'PointNo', 'x', 'y'])[['ImageName', 'PointNo', 'x', 'y']]

        # Mit Deklaration als multiframe erfolgt Zugriff auf einzelne rows immer
        # mit einer Doppelangabe imagename + pointno
        self.csvData = data.set_index(['ImageName', 'PointNo'])
        self.csvData.sort_index(inplace=True)
        self.imageNames = self.csvData.index.levels[0]  # gibt die Dateinamen als Liste zurück
        # print(self.imageNames)

    '''====================================================================='''

    def extractThrombusPositions(self, image_name):
        number_of_thrombi = len(self.csvData.loc[image_name])

        if number_of_thrombi == 1:
            if self.csvData.loc[(image_name, 0)]['x'].item() > 0:
                return [(0, 0)]  # means: sequence is classified as thrombus free
            else:
                return [(0, 1)]  # means: sequence was not classified by radiologist

        positions = list()
        for i in range(1, number_of_thrombi):
            positions.append((self.csvData.loc[(image_name, i)]['x'],
                              self.csvData.loc[(image_name, i)]['y']))

        return positions

    '''====================================================================='''

    def getNumberOfEntries(self):
        return len(self.imageNames)

    '''=============Möglichkeiten Zugriff auf pandas Dataframe=============='''
    # for name in self.imageNames:
    #    print(name)
    # print(self.csvData)
    # print(self.csvData.index)
    # print(len(self.csvData.loc['002-02-aci-r-f'])) # Ermittelt Anzahl der Thromben: 1 =
    # print(self.csvData.loc[('001-01-aci-r-s', 0)]['x']) # Zugriff auf x-Koordinate

    '''
    # 1. Zugriffsmöglichkeit mit Zuweisung des ImageNames als Index:
    self.csvData = data.set_index('ImageName')
    selected = self.csvData.loc['001-01-aci-r-f']
    print(selected)
    print(selected.iloc[0])
    print(selected.iloc[0]['x'])
    print(selected.iloc[0]['y'])
    print(selected.iloc[1]['x'])
    print(selected.iloc[1]['y'])

    # 2. Zugriffsmöglichkeit ohne Zuweisung des ImageNames als Index:
    selected = self.csvData.loc[self.csvData['ImageName'] == '001-01-aci-r-s']
    print(selected)
    print(selected.iloc[0]) # liefert einzelne row/Serie zurück mit x,y-Werten
    print(selected.iloc[1]) # liefert einzelne row/Serie zurück mit x,y-Werten
    print(selected.iloc[0]['x']) # x-Koordinate des Thrombus
    print(selected.iloc[0]['y'])
    print(selected.iloc[1]['x'])
    print(selected.iloc[1]['y'])
    '''
    # selected.shape returns (n_rows, n_columns)
    # selected.empty returns True if DataFrame has no entry
