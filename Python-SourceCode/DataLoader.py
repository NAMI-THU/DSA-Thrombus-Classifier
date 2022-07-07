# -*- coding: utf-8 -*-
"""
@author: mittmann
"""

import os
import numpy as np
import nibabel
import pandas as pd
from ImageUtils import ImageUtils

class DataLoader(object):
    def __init__(self, data_path="", csv_path=""):
        self.dataPath = data_path
        self.csvPath = csv_path
        self.csvData = None         #is a multiindex dataframe containing all csv data
        self.imageNames = list()
        self.currentIndex = -1 # Standard: -1, für Entwicklungszwecke > 0
        self.files = os.listdir(self.dataPath)
        self.frontalAndLaterialViewLastIndex = False

    '''====================================================================='''
    def getNextEntry(self):
        print("getNextExtry")
        if len(self.imageNames) == 0:
            raise RuntimeError("Cannot get next entry. \
                                The csv-file has to be read first.")
            return
        
        if self.frontalAndLaterialViewLastIndex == True:
            self.currentIndex += 2
        else:
            self.currentIndex += 1

        #Stellt sicher, dass keine out-of-index-error auftritt:
        if self.currentIndex >= len(self.imageNames):
            return None, None, None, None

        #for index, filename in enumerate(files):
        filename = self.imageNames[self.currentIndex] + ".nii"
        dsa_image_path = os.path.join(self.dataPath, filename)
        dsa_image = nibabel.load(dsa_image_path)

        image_data = dsa_image.get_fdata()
        image_data = ImageUtils.fillBlackBorderWithRandomNoise(image_data, 3050)

        positions = self.extractThrombusPositions(self.imageNames[self.currentIndex])
        print(positions)
        
        # image other view:
        #check, if other view exists:
        if (self.currentIndex + 1) >= len(self.imageNames):
            self.frontalAndLaterialViewLastIndex = False
            image_data2 = None
            positions2 = None
            return image_data, image_data2, positions, positions2

        filename2 = self.imageNames[self.currentIndex + 1] + ".nii"

        print(filename + " | " + filename2)
        print(filename2[:7] in filename)

        if (filename2[:7] in filename) == False:
            # means: second image does not belong to the actual image series,
            # as the files do not have the same filename starting with.
            self.frontalAndLaterialViewLastIndex = False
            image_data2 = None
            positions2 = None
            return image_data, image_data2, positions, positions2

        # means: second image belongs to actual image series. So, load it and
        #        return it.
        self.frontalAndLaterialViewLastIndex = True
        dsa_image_path2 = os.path.join(self.dataPath, filename2)
        dsa_image2 = nibabel.load(dsa_image_path2)

        image_data2 = dsa_image2.get_fdata()
        image_data2 = ImageUtils.fillBlackBorderWithRandomNoise(image_data2, 3050)
        
        positions2 = self.extractThrombusPositions(self.imageNames[self.currentIndex + 1])
        print(positions2)
        
        return image_data, image_data2, positions, positions2

    '''====================================================================='''
    def getCurrentIndex(self):
        return self.currentIndex
    '''====================================================================='''
    def getFrontalAndLateralViewLastIndex(self):
        return self.frontalAndLaterialViewLastIndex
    '''====================================================================='''
    def loadCsvData(self):
        name = "Dataset_1.csv"
        path = os.path.join(self.csvPath, name)

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
        self.imageNames = self.csvData.index.levels[0] # gibt die Dateinamen als Liste zurück
        #for name in self.imageNames:
        #    print(name)
        #print(self.csvData)
        #print(self.csvData.index)
        #print(len(self.csvData.loc['002-02-aci-r-f'])) # Ermittelt Anzahl der Thromben: 1 = 
        #print(self.csvData.loc[('001-01-aci-r-s', 0)]['x']) # Zugriff auf x-Koordinate

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

    '''====================================================================='''
    def extractThrombusPositions(self, image_name):
        number_of_thrombi = len(self.csvData.loc[image_name])
        if number_of_thrombi == 1:
            if self.csvData.loc[(image_name, 0)]['x'] > 0:
                return [(0, 0)] # means: sequence is classified as thrombus free
            else:
                return [(-1, 0)] # means: sequence was not classified by radiologist
        
        positions = list()
        for i in range(1, number_of_thrombi):
            positions.append((self.csvData.loc[(image_name, i)]['x'],
                              self.csvData.loc[(image_name, i)]['y']))
        
        return positions
     
    '''====================================================================='''
    def getNumberOfEntries(self):
        return len(self.imageNames)
            
