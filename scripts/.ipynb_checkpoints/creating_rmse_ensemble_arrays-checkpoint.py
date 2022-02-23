#!/bin/env python

# This script returns the RMSE between timeseries of each output. Taking in 8 outputs results in 64 RMSE outputs. 

import timeit
import shutil
import os
import numpy as np
from tqdm import tqdm
from scipy.stats.stats import pearsonr
from src.helper_functions import multi_apply_along_axis
from src.helper_functions import RMSE
import sys

variable = sys.argv[1]

input_path = '/gws/nopw/j04/lancs_atmos/users/carter10/Antarctica_Climate_Variability/Ensemble_Array_Data/'
temporary_destination_path = f'/work/scratch-pw/carter10/creating_correlation_ensemble_arrays_{variable}'
final_destination_path = '/gws/nopw/j04/lancs_atmos/users/carter10/Antarctica_Climate_Variability/Ensemble_Array_Data/'

os.mkdir(temporary_destination_path) #Creating temporary directory to save results to (will delete this dir later)

time = timeit.default_timer()

ensemble_stl_decomp = np.load(f'{temporary_destination_path}/ensemble_stl_decomp_{variable}.npy')
# Note ensemble_stl_decomp has shape [8,3,456,392,504] corresponding to [model,component,time,glat,glon]
ensemble_stl_decomp_bias_adjusted = ensemble_stl_decomp.copy()
ensemble_stl_decomp_bias_seasonal_adjusted = ensemble_stl_decomp.copy()
ensemble_stl_decomp_bias_seasonal_residual_adjusted = ensemble_stl_decomp.copy()

for i in tqdm(np.arange(0,8,1)):
    bias = ensemble_stl_decomp[i,0]-ensemble_stl_decomp.mean(0)[0]
    seasonal_std_bias = ensemble_stl_decomp[i,1].std(0)/ensemble_stl_decomp.mean(0)[1].std(0)
    residual_std_bias = ensemble_stl_decomp[i,2].std(0)/ensemble_stl_decomp.mean(0)[2].std(0)

    ensemble_stl_decomp_bias_adjusted[i]=ensemble_stl_decomp[i]-bias
    ensemble_stl_decomp_bias_seasonal_adjusted[i]=ensemble_stl_decomp[i,0]-bias+ensemble_stl_decomp[i,1]/seasonal_std_bias+ensemble_stl_decomp[i,2]
    ensemble_stl_decomp_bias_seasonal_residual_adjusted[i]=ensemble_stl_decomp[i,0]-bias+ensemble_stl_decomp[i,1]/seasonal_std_bias+ensemble_stl_decomp[i,2]/residual_std_bias

rmse_ensemble = []
rmse_ensemble_bias_adjusted = []
rmse_ensemble_bias_seasonal_adjusted = []
rmse_ensemble_bias_seasonal_residual_adjusted = []

for i in tqdm(np.arange(0,8,1)):
    for j in tqdm(np.arange(0,8,1)):
        rmse = multi_apply_along_axis(RMSE, 1, [ensemble_stl_decomp[i],ensemble_stl_decomp[j]]) #returns shape [3,392,504]
        rmse_bias_adjusted = multi_apply_along_axis(RMSE, 1, [ensemble_stl_decomp_bias_adjusted[i],ensemble_stl_decomp_bias_adjusted[j]]) #returns shape [3,392,504]
        rmse_bias_seasonal_adjusted = multi_apply_along_axis(RMSE, 1, [ensemble_stl_decomp_bias_seasonal_adjusted[i],ensemble_stl_decomp_bias_seasonal_adjusted[j]]) #returns shape [3,392,504]
        rmse_bias_seasonal_residual_adjusted = multi_apply_along_axis(RMSE, 1, [ensemble_stl_decomp_bias_seasonal_residual_adjusted[i],ensemble_stl_decomp_bias_seasonal_residual_adjusted[j]]) #returns shape [3,392,504]
        rmse_ensemble.append(rmse)
        rmse_ensemble_bias_adjusted.append(rmse_bias_adjusted)
        rmse_ensemble_bias_seasonal_adjusted.append(rmse_bias_seasonal_adjusted)
        rmse_ensemble_bias_seasonal_residual_adjusted.append(rmse_bias_seasonal_residual_adjusted)

np.save(f'{temporary_destination_path}/ensemble_stl_decomp_rmse_{variable}',np.array(rmse_ensemble))
np.save(f'{temporary_destination_path}/ensemble_stl_decomp_rmse_bias_adjusted_{variable}',np.array(rmse_ensemble_bias_adjusted))
np.save(f'{temporary_destination_path}/ensemble_stl_decomp_rmse_bias_seasonal_adjusted_{variable}',np.array(rmse_ensemble_bias_seasonal_adjusted))
np.save(f'{temporary_destination_path}/ensemble_stl_decomp_rmse_bias_seasonal_residual_adjusted_{variable}',np.array(rmse_ensemble_bias_seasonal_residual_adjusted))

for file_name in os.listdir(temporary_destination_path):
    shutil.move(f'{temporary_destination_path}/{file_name}', f'{final_destination_path}{file_name}')    
    
os.rmdir(temporary_destination_path) 

print(f'Total Time Taken ({variable}:', timeit.default_timer() - time)


