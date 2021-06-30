# net.eval() - podivat se co to znamena
import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset

import gc
import time

import cv2 
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from datetime import datetime
from tqdm import tqdm

def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # this file path
    dirname = '/home/anya/Documents/kits19/pytorch/Pytorch-UNet'
    modelsFolder = 'models'
    datasetFolder = 'data/test_npz'
    outputFolder = 'predictionstest/case_00168'

    modelsDir = os.path.join(dirname, modelsFolder)
    pathToImages = os.path.join(dirname, datasetFolder)          # Images


    # load cases
    cases = os.listdir(pathToImages)
    cases.sort()
    cases = cases[:1] # cases[:casesnum]

    inputHeight = 512
    inputWidth = 512

    # load model
    modelPaths = os.listdir(modelsDir)
    modelPaths.sort()

    for modelPath in modelPaths[4:5]:
        outputDir = os.path.join(modelsDir, modelPath , outputFolder)
        bestModelDir = os.path.join(modelsDir, modelPath, 'checkpoints')
        files = os.listdir(bestModelDir)
        files = [os.path.join(bestModelDir, f) for f in files] # add path to each file
        files.sort(key=lambda x: os.path.getmtime(x))
        bestModel = files[-1:]

        net = UNet(n_channels=1, n_classes=1)

        logging.info("Loading model {}".format(bestModel[0]))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device {device}')
        net.to(device=device)
        net.load_state_dict(torch.load(bestModel[0], map_location=device))
        net.eval()
        logging.info("Model loaded !")

        for case in cases:
            # load data
            imagePath = os.path.join(pathToImages, case)
            # load img with stopwatch for info
            stopwatch = time.time()
            images = np.load(imagePath, None, True)
            print(f'case {case} was loaded in {time.time() - stopwatch} seconds')

            keys = images.files
            slicenum = 40 
            keys = keys[:slicenum] # take all -> files[:] or take 10 for example -> files[:10]
            # predict
            p = {}  # predictions
            for key in keys:
                X = images[key][0]

                # # Histograms Equalization
                # X = cv2.equalizeHist(np.uint8(X))

                # # create a CLAHE object (Arguments are optional).
                # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                # X = clahe.apply(np.uint8(X))

                X = torch.from_numpy(X)
                X = X.to(device, dtype=torch.float32)
                X = X.unsqueeze(0).unsqueeze(0)
                prediction = net(X)
                prediction = prediction.cpu().detach().numpy()
                prediction = (prediction > 0) * 255.0
                prediction = prediction.astype(np.uint8).reshape(512, 512)

                os.makedirs(os.path.join(outputDir), exist_ok=True)    
                # to save in png       
                image_path = os.path.join(outputDir, f'{key}_prediction.png')
                cv2.imwrite(image_path, prediction)

                p[f'{key}'] = prediction.reshape(1, 512, 512).astype(np.float32)
            # to save as npz for evaluation
            print(f'Saving was started. This operation can take a while, be patient please...')
            pPath = os.path.join(outputDir, f'prediction_{case}')
            # np.savez_compressed(pPath, **p)
            print(f'Array with {len(p)} images was saved.')
print("Done!")
