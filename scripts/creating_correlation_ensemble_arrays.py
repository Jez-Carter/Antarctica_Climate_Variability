#!/bin/env python

# This script returns the pearson linear correlation between timeseries of each output. Taking in 8 outputs results in 64 correlation outputs. 

import timeit
import shutil
import os
import numpy as np
from tqdm import tqdm
from scipy.stats.stats import pearsonr
from src.helper_functions import multi_apply_along_axis
import sys

variable = sys.argv[1]

input_path = '/gws/nopw/j04/lancs_atmos/users/carter10/Antarctica_Climate_Variability/Ensemble_Array_Data/'
temporary_destination_path = f'/work/scratch-pw/carter10/creating_correlation_ensemble_arrays_{variable}'
final_destination_path = '/gws/nopw/j04/lancs_atmos/users/carter10/Antarctica_Climate_Variability/Ensemble_Array_Data/'

os.mkdir(temporary_destination_path) #Creating temporary directory to save results to (will delete this dir later)

time = timeit.default_timer()

ensemble_stl_decomp = np.load(f'{input_path}ensemble_stl_decomp_{variable}.npy')
# Note ensemble_stl_decomp has shape [8,3,456,392,504] corresponding to [model,component,time,glat,glon]
corr_ensemble = []
for i in tqdm(np.arange(0,8,1)):
    for j in tqdm(np.arange(0,8,1)):
        corr = multi_apply_along_axis(pearsonr, 1, [ensemble_stl_decomp[i],ensemble_stl_decomp[j]]) #returns shape [3,2,392,504]
        corr_ensemble.append(corr)
np.save(f'{temporary_destination_path}/ensemble_stl_decomp_corr_{variable}',np.array(corr_ensemble))

for file_name in os.listdir(temporary_destination_path):
    shutil.move(f'{temporary_destination_path}/{file_name}', f'{final_destination_path}{file_name}')    
    
os.rmdir(temporary_destination_path) 

print(f'Total Time Taken ({variable}:', timeit.default_timer() - time)
