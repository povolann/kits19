# Convert .nii.gz to .npz

import os
import gc
import time
import numpy as np
import nibabel as nib

# this file path
dirname = os.path.dirname(__file__)

# constants
dataFolder = 'test'
outFolder = 'bordel'
imgCnt = 10

dataPath = os.path.join(dirname, dataFolder)
outPath = os.path.join(dirname, outFolder)

# create output directory
os.makedirs(outPath, exist_ok=True)


# load cases
cases = os.listdir(dataPath) # 209 scans which are anotated
cases.sort()
cases = cases[:] # cases[:imgCnt]

for case in cases:
    imagePath = os.path.join(dataPath, case, 'imaging.nii.gz')

    # load img with stopwatch for info
    stopwatch = time.time()

    imgarr = nib.load(imagePath).get_data().astype(np.uint8)

    print(f'case {case} was loaded in {time.time() - stopwatch} seconds')
    print(f'* img has minimum value: {np.min(imgarr)} and maximum value: {np.max(imgarr)}')
    # output dictionaries
    x = {}    # images

    for n in range(imgarr.shape[0]):
        img = imgarr[n,:,:] 
        img = np.multiply(img, 1.0/255.0, dtype=np.float32)

        # store in dictionary
        x[f'{case}_{n}'] = img.reshape(1, imgarr.shape[1], imgarr.shape[2]).astype(np.float32)
        stopwatch = time.time()
        pass
    print(f'Saving was started. This operation can take a while, be patient please...')
    xPath = os.path.join(outPath, f'{case}'+'.npz')
    np.savez_compressed(xPath, **x)
    print(f'Array with {len(x)} images was saved in {time.time() - stopwatch} seconds')
    del x
    pass
pass

print(f'Done!')