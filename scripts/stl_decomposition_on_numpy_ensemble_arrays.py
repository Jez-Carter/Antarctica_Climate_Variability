#!/bin/env python

if __name__ ==  '__main__':

    # This script applies the stl decomposition method to the ensemble of monthly timeseriesover 40 years.

    import sys
    import timeit
    import shutil
    import os
    import numpy as np
    import dask.array as da
    from src.helper_functions import stl_decomposition
    
    variable = sys.argv[1]

    start_date = '1-1-1981'
    frequency='M'

    input_path = '/gws/nopw/j04/lancs_atmos/users/carter10/Antarctica_Climate_Variability/Ensemble_Array_Data/'
    temporary_destination_path = f'/work/scratch-pw/carter10/stl_decomposition_{variable}'
    final_destination_path = '/gws/nopw/j04/lancs_atmos/users/carter10/Antarctica_Climate_Variability/Ensemble_Array_Data/'

    os.mkdir(temporary_destination_path) #Creating temporary directory to save results to (will delete this dir later)

    ensemble = np.load(f'{input_path}ensemble_{variable}.npy').astype('float32')
    ensemble_dask = da.from_array(ensemble, chunks=(8,456, 28,28))  # The np array has shape (8,456,392,504), which is split into 14*18 (8,456, 28,28) chunks 
    input_data = ensemble_dask[:,:,:,:]
    dtype = input_data.dtype
    output_shape = (3,456)
    ensemble_stl_decomp_dask = da.apply_along_axis(stl_decomposition, axis=1, arr=input_data, start_date=start_date,frequency=frequency, dtype=dtype, shape = output_shape)
    ensemble_stl_decomp = ensemble_stl_decomp_dask.compute(scheduler = 'processes')
    np.save(f'{temporary_destination_path}/ensemble_stl_decomp_{variable}.npy',ensemble_stl_decomp)

    for file_name in os.listdir(temporary_destination_path):
        shutil.move(f'{temporary_destination_path}/{file_name}', f'{final_destination_path}{file_name}')

    os.rmdir(temporary_destination_path) 