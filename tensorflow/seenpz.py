import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PIL import Image

# this file path
dirname = os.path.dirname(__file__)

# constants
dataFolder = 'npz/test'
outFolder = 'testpng'
imgCnt = 1

dataPath = os.path.join(dirname, dataFolder)
outPath = os.path.join(dirname, outFolder)

# create output directory
os.makedirs(outPath, exist_ok=True)

# load cases
cases = os.listdir(dataPath)
cases.sort()
# cases = cases[:imgCnt]

for case in cases:
    imagePath = os.path.join(dataPath, case)

    # load img with stopwatch for info
    images = np.load(imagePath, None, True)
    keys = images.files
    slicenum = 2
    keys = keys[:slicenum]
    for n in keys:
        # image convert
        slice = np.array(images[n])
        slice = slice * 255.0
        slice = slice.astype(np.uint8).reshape(512, 512)
        image = Image.fromarray(slice)          
        
        # image save
        image.save(os.path.join(outPath, f"{n}.png"))
        pass
    pass


print('Done!')