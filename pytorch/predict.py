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


# this file path
dirname = os.path.dirname(__file__)
modelsFolder = 'models'
datasetFolder = 'data/test'
outputFolder = 'predictionstest'

modelsDir = os.path.join(dirname, modelsFolder)
pathToImages = os.path.join(dirname, datasetFolder)          # Images

# load cases
cases = os.listdir(pathToImages)
cases.sort()
cases = cases[:1] # cases[:casesnum], for 180 32:33

inputHeight = 512
inputWidth = 512

# load model
modelPaths = os.listdir(modelsDir)
modelPaths.sort()

for modelPath in modelPaths[-2:]:
    outputDir = os.path.join(modelsDir, modelPath , outputFolder)
    bestModelDir = os.path.join(modelsDir, modelPath, 'checkpoints/')
    os.chdir(bestModelDir)
    files = filter(os.path.isfile, os.listdir(bestModelDir))
    files = [os.path.join(bestModelDir, f) for f in files] # add path to each file
    files.sort(key=lambda x: os.path.getmtime(x))
    bestModel = files[-1:]
    print(bestModel[0])

    for case in cases:
        # load data
        imagePath = os.path.join(pathToImages, case)
        # load img with stopwatch for info
        stopwatch = time.time()
        images = np.load(imagePath, None, True)
        print(f'case {case} was loaded in {time.time() - stopwatch} seconds')

        keys = images.files
        slicenum = 40
        keys = keys[:10] # take all -> files[:] or take 10 for example -> files[:10]
        try:
            # predict
            p = {}  # predictions
            for key in keys:
                X = images[key]
                
                prediction = predict_img




def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=1)

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    args = get_args()
    in_files = args.input
    out_files = get_output_filenames(args)

    net = UNet(n_channels=1, n_classes=1)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    for i, fn in enumerate(in_files):
        logging.info("\nPredicting image {} ...".format(fn))

        img = Image.open(fn)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            out_fn = out_files[i]
            result = mask_to_image(mask)
            result.save(out_files[i])

            logging.info("Mask saved to {}".format(out_files[i]))

        if args.viz:
            logging.info("Visualizing results for image {}, close to continue ...".format(fn))
            plot_img_and_mask(img, mask)
