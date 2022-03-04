#!/bin/env python

# This script takes the ensemble stl decomposition array of shape [8,3,456,392,504] and samples it to a grid cell on Larsen C [235,55]

import numpy as np 
import os 
import shutil

input_path = '/home/users/carter10/Shared_Storage/Antarctica_Climate_Variability/Ensemble_Array_Data/'
temporary_destination_path = f'/work/scratch-pw/carter10/sampling_stl_ensemble_arrays_larsenc'
final_destination_path = '/gws/nopw/j04/lancs_atmos/users/carter10/Antarctica_Climate_Variability/Ensemble_Array_Data/'

os.mkdir(temporary_destination_path) #Creating temporary directory to save results to (will delete this dir later)

variables = ['snowfall','temperature','melt']

for variable in variables:
    data = np.load(f'{input_path}ensemble_stl_decomp_{variable}.npy') #has shape [8,3,456,392,504]
    data = data[:,:,:,235,55] # filtering to Larsen C grid cell
    np.save(f'{temporary_destination_path}/ensemble_stl_decomp_{variable}_larsen_c',data)

for file_name in os.listdir(temporary_destination_path):
    shutil.move(f'{temporary_destination_path}/{file_name}', f'{final_destination_path}{file_name}')    
    
os.rmdir(temporary_destination_path) 