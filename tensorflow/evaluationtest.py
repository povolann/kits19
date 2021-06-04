# located in starter_code
# from starter_code.utils import load_segmentation
from utils import load_segmentation
import os

import numpy as np
import nibabel as nib

# this file path
dirname = os.path.dirname(__file__)
homeFolder = '/home/anya/Documents/kits19/tensorflow/'
datasetFolder = '/home/anya/Documents/kits19/data' # evaluating the test data

modelsFolder = 'models'
modelsDir = os.path.join(homeFolder, modelsFolder)
modelPaths = os.listdir(modelsDir)
modelPaths.sort()
modelNum = -2
modelPaths = modelPaths[-1:]
#modelPaths = modelPaths[modelNum:modelNum+1] # for -1 need to edit, only 1 model evalutaion

predictionsFolder = os.path.join(modelsDir, modelPaths[0], 'predictionstest')

def evaluate(case_id, predictions):
    # Handle case of softmax output
    if len(predictions.shape) == 4:
        predictions = np.argmax(predictions, axis=-1)

    # Check predictions for type and dimensions
    if not isinstance(predictions, (np.ndarray, nib.Nifti1Image)):
        raise ValueError("Predictions must by a numpy array or Nifti1Image")
    if isinstance(predictions, nib.Nifti1Image):
        predictions = predictions.get_data()

    if not np.issubdtype(predictions.dtype, np.integer):
        predictions = np.round(predictions)
    predictions = predictions.astype(np.uint8)    

    # Load ground truth segmentation
    gt = load_segmentation(case_id).get_data()
    
    # Make sure shape agrees with case
    if not predictions.shape == gt.shape:
        raise ValueError(
            ("Predictions for case {} have shape {} "
            "which do not match ground truth shape of {}").format(
                case_id, predictions.shape, gt.shape
            )
        )

    try:
        # Compute tumor+kidney Dice
        tk_pd = np.greater(predictions, 0)
        tk_gt = np.greater(gt, 0)
        tk_dice = 2*np.logical_and(tk_pd, tk_gt).sum()/(
            tk_pd.sum() + tk_gt.sum()
        )
    except ZeroDivisionError:
        return 0.0, 0.0

    try:
        # Compute tumor Dice
        tu_pd = np.greater(predictions, 1)
        tu_gt = np.greater(gt, 1)
        tu_dice = 2*np.logical_and(tu_pd, tu_gt).sum()/(
            tu_pd.sum() + tu_gt.sum()
        )
    except ZeroDivisionError:
        return tk_dice, 0.0

    return tk_dice #, tu_dice

# load cases
cases = os.listdir(datasetFolder)
cases.sort()
cases = cases[:]
tk_dice_all = []

for case in cases:
    # select data in to single array
    pred = [] # predictions
    case_id = case

    predictionsPath = os.path.join(predictionsFolder, f'prediction_{case}.npz')
    predictions = np.load(predictionsPath, None, True)

    for slice in predictions.files:
        predarr = np.array(predictions[slice])
        # predarr = predarr.reshape((1, 512, 512, 1)) no need to reshape
        pred.append(predarr)
        pass
    
    # concatenate predictions
    P = np.concatenate(pred)

    tk_dice = evaluate(case_id, P)

    fileDir = os.path.join(modelsDir, modelPaths[0], 'parameters')
    file = open(fileDir, "a")
    L1 = [f'Tumor+kidney Dice for {case}: {tk_dice}\n'] 
    file.writelines(L1)
    file.close()

    print(f'tumor+kidney Dice for {case}: {tk_dice}')

    # Calculate the avarage mean
    tk_dice_all.append(tk_dice)

tk_dice_avg = sum(tk_dice_all)/len(cases)

fileDir = os.path.join(modelsDir, modelPaths[0], 'parameters')
file = open(fileDir, "a")
L2 = [f'Tumor+kidney Dice for all cases from model {modelPaths[0]}: {tk_dice_avg}'] 
file.writelines(L2)
file.close()

print(f'tumor+kidney Dice for all cases from model {modelPaths[0]}: {tk_dice_avg}')
    
print('Done!')
