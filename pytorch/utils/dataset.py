from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
import gc
import time
from tqdm import tqdm
from PIL import Image
import cv2


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix=''):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir

        self.scale = scale
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.X = None
        self.Y = None
        img = np.load(imgs_dir, None, True)
        self._get_data(imgs_dir, masks_dir)

    def _get_data(self, imgs_dir, masks_dir):
        stopwatch = time.time()
        img = np.load(imgs_dir, None, True)
        mask = np.load(masks_dir, None, True)

        startImage = 0
        cntOfImagesFromDataset = 1000 # 11956           # how many slides are used for training
        endImage = startImage + cntOfImagesFromDataset

        cherryPicking = True                             # Pick only valid images from dataset
        cherryMin = 0.01                                 # used only if cherryPicking is True
        cherryMax = 1.00                                 # used only if cherryPicking is True

        keys = img.files
        keys = keys[startImage:endImage] # take all -> files[:] or take 10 for example -> files[:10]

        # select data in to single array
        x = [] # images
        y = [] # masks
        for file in tqdm(keys): 
            xarr = np.array(img[file])
            yarr = np.array(mask[file])

            assert xarr.size == yarr.size, \
            f'Image and mask {file} should be the same size, but are {xarr.size} and {yarr.size}'

            # cherry picking
            if cherryPicking:
                sum = float(yarr.sum())
                area = float(yarr.size)
                sumratio = sum / area

                if sumratio < cherryMin or sumratio > cherryMax:
                    continue

            xarr = self.preprocess(xarr, self.scale)
            yarr = self.preprocess(yarr, self.scale)

            x.append(xarr)
            y.append(yarr)

        print(f"Concatenation ...")
        self.X = np.concatenate(x)
        self.Y = np.concatenate(y)

        logging.info(f'Creating dataset with and imgs = {len(self.X)} examples')
        logging.info(f'Data loaded in {time.time() - stopwatch} seconds')
        
    def __len__(self):
        return len(self.X) # len(dataset) = 5088 for Caravana, 8646 for kits19

    @classmethod
    def preprocess(cls, np_img, scale):
        pil_img = Image.fromarray(np.uint8(np_img[0] * 255) , 'L')
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_n = np.array(pil_img)

        # Histograms Equalization
        img_eq = cv2.equalizeHist(img_n)

        # # create a CLAHE object (Arguments are optional).
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # img_cl = clahe.apply(img_n)
        # cv2.imwrite('equalized.png', img_eq)
        # cv2.imwrite('clahe.png', img_cl)
        # cv2.imwrite('normal.png', img_nd)

        img_nd = img_eq

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    # load i-th image
    def __getitem__(self, i):
        X = self.X[i]
        Y = self.Y[i]

        return {
            'image': torch.from_numpy(X).type(torch.FloatTensor),
            'mask': torch.from_numpy(Y).type(torch.FloatTensor)
        }

class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')

class kits19(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='')
