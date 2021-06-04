# Prediction with chosen models
# Save as npz file or png images
# TODO: you can merge heatmap of prediction and with mask to better comparision

import os
import gc
import time

import numpy as np
import cv2 
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import keras

from keras.models import load_model
from keras import backend as K
from efficientnet.tfkeras import EfficientNetB4 # needed for training with efficientnet

from datetime import datetime
from tqdm import tqdm


# this file path
dirname = os.path.dirname(__file__)
modelsFolder = 'models'
datasetFolder = 'npz/test'
outputFolder = 'predictionstest'

bestModelName = 'model_best.h5'

modelsDir = os.path.join(dirname, modelsFolder)              # Models
pathToImages = os.path.join(dirname, datasetFolder)          # Images

# load cases
cases = os.listdir(pathToImages)
cases.sort()
cases = cases[:] # cases[:casesnum]

inputHeight = 512
inputWidth = 512

fake3channels = True # True if your model expect ?:?:3

# load model
modelPaths = os.listdir(modelsDir)
modelPaths.sort()

# choosing the model
for modelPath in modelPaths[18:19]:
    outputDir = os.path.join(modelsDir, modelPath , outputFolder)
    bestModelPath = os.path.join(modelsDir, modelPath, bestModelName)
    if not os.path.exists(bestModelPath):
        continue

    # load model
    model = load_model(bestModelPath, compile=False)
    
    for case in cases:
        # load data
        imagePath = os.path.join(pathToImages, case)
        # load img with stopwatch for info
        stopwatch = time.time()
        images = np.load(imagePath, None, True)
        print(f'case {case} was loaded in {time.time() - stopwatch} seconds')

        keys = images.files
        keys = keys[:] # take all -> files[:] or take 10 for example -> files[:10]
        try:
            # predict
            p = {}  # predictions
            for key in keys:
                X = images[key]
                if fake3channels:
                    X = X.reshape(1, inputHeight, inputWidth, 1)
                    X = np.concatenate((X, X, X), axis=3)
                    pass
                
                prediction = model.predict(X)
                prediction = np.round(prediction) * 255.0
                prediction = prediction.astype(np.uint8).reshape(512, 512)

                os.makedirs(os.path.join(outputDir), exist_ok=True)    
                # to save in png       
                # image_path = os.path.join(outputDir, f'{key}_prediction.png')
                # cv2.imwrite(image_path, prediction)
                p[f'{key}'] = prediction.reshape(1, 512, 512).astype(np.float32)
                pass
            # to save as npz for evaluation
            print(f'Saving was started. This operation can take a while, be patient please...')
            pPath = os.path.join(outputDir, f'prediction_{case}')
            np.savez_compressed(pPath, **p)
            print(f'Array with {len(p)} images was saved.')
            del p
            pass
        except:
            print(f'Model fall to exception: {modelPath}')
            continue
        pass
    pass
pass

print('Done!')
