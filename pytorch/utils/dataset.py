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


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix=''):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        # self.ids = 
        

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    # data loading
    def __getitem__(self, i): 
        stopwatch = time.time()
        mask = np.load(self.masks_dir, None, True)
        img = np.load(self.imgs_dir, None, True)
        startImage = 0
        cntOfImagesFromDataset = 10000 # 16220           # you can use large number for all images like 999999
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
            pass

        # concatenate selected images and masks
        print(f"Concatenation ...")
        X = np.concatenate(x)
        Y = np.concatenate(y)

        logging.info(f'Creating dataset with {X.shape[0]} examples')

        # force clean up memory
        del x
        del y
        del img
        del mask
        gc.collect()

        logging.info(f'Data loaded in {time.time() - stopwatch} seconds')

        return {
            'image': torch.from_numpy(X).type(torch.FloatTensor),
            'mask': torch.from_numpy(Y).type(torch.FloatTensor)
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')
