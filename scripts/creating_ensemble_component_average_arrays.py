#!/bin/env python

# This script returns the mean melt for the ensemble of outputs. Takes in [8,3,456,392,504] returns [392,504]

import numpy as np
import shutil
import os
import sys

variable = sys.argv[1]

input_path = '/gws/nopw/j04/lancs_atmos/users/carter10/Antarctica_Climate_Variability/Ensemble_Array_Data/'
temporary_destination_path = f'/work/scratch-pw/carter10/creating_ensemble_average_statistics_{variable}'
final_destination_path = '/gws/nopw/j04/lancs_atmos/users/carter10/Antarctica_Climate_Variability/Ensemble_Array_Data/'

os.mkdir(temporary_destination_path) #Creating temporary directory to save results to (will delete this dir later)

ensemble_stl_decomp = np.load(f'{input_path}ensemble_stl_decomp_{variable}.npy')#,mmap_mode = 'r')#.astype('float32') # returns shape [8,3,456,392,504]
ensemble_stl_decomp_mean = ensemble_stl_decomp[:,0].mean(axis=1) #selecting trend component and taking mean along time axis, returns shape [8,392,504]
ensemble_stl_decomp_trend_stddev = ensemble_stl_decomp[:,0].std(axis=1) #selecting trend component and taking stddev along time axis, returns shape [8,392,504]
ensemble_stl_decomp_seasonal_stddev = ensemble_stl_decomp[:,1].std(axis=1) #selecting seasonal component and taking stddev along time axis, returns shape [8,392,504]
ensemble_stl_decomp_residual_stddev = ensemble_stl_decomp[:,2].std(axis=1) #selecting residual component and taking stddev along time axis, returns shape [8,392,504]
np.save(f'{temporary_destination_path}/ensemble_stl_decomp_{variable}_mean',ensemble_stl_decomp_mean)
np.save(f'{temporary_destination_path}/ensemble_stl_decomp_{variable}_trend_stddev',ensemble_stl_decomp_trend_stddev)
np.save(f'{temporary_destination_path}/ensemble_stl_decomp_{variable}_seasonal_stddev',ensemble_stl_decomp_seasonal_stddev)
np.save(f'{temporary_destination_path}/ensemble_stl_decomp_{variable}_residual_stddev',ensemble_stl_decomp_residual_stddev)

for file_name in os.listdir(temporary_destination_path):
    shutil.move(f'{temporary_destination_path}/{file_name}', f'{final_destination_path}{file_name}')    
    
os.rmdir(temporary_destination_path) 
