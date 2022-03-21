#!/bin/env python

# This script returns the mean melt for the ensemble of outputs. Takes in [8,3,456,392,504] returns [392,504]

import numpy as np
import shutil
import os

input_path = '/gws/nopw/j04/lancs_atmos/users/carter10/Antarctica_Climate_Variability/Ensemble_Array_Data/'
temporary_destination_path = f'/work/scratch-pw/carter10/mean_melt'
final_destination_path = '/gws/nopw/j04/lancs_atmos/users/carter10/Antarctica_Climate_Variability/Ensemble_Array_Data/'

os.mkdir(temporary_destination_path) #Creating temporary directory to save results to (will delete this dir later)

ensemble_melt = np.load(f'{input_path}ensemble_melt.npy') # returns shape [8,456,392,504]
mean_melt = ensemble_melt.mean((0,1)) # taking mean along time axis and model axis

np.save(f'{temporary_destination_path}mean_melt',mean_melt)

for file_name in os.listdir(temporary_destination_path):
    shutil.move(f'{temporary_destination_path}/{file_name}', f'{final_destination_path}{file_name}')    
    
os.rmdir(temporary_destination_path) 