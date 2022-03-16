#!/bin/env python

# This script returns the mean melt for the ensemble of outputs. Takes in [8,3,456,392,504] returns [392,504]

import numpy as np
import numpy.ma as ma
import shutil
import os
import sys
import iris

variable = sys.argv[1]

input_path = '/gws/nopw/j04/lancs_atmos/users/carter10/Antarctica_Climate_Variability/Ensemble_Array_Data/'
temporary_destination_path = f'/work/scratch-pw/carter10/creating_ensemble_ice_sheet_only_totals_{variable}'
final_destination_path = '/gws/nopw/j04/lancs_atmos/users/carter10/Antarctica_Climate_Variability/Ensemble_Array_Data/'
mask_path = '/gws/nopw/j04/lancs_atmos/users/carter10/Antarctica_Climate_Variability/Mask_Data/'
grid_cube_path = '/home/users/carter10/Shared_Storage/Antarctica_Climate_Variability/Postprocessed_Data/MetUM_011_ERA_INT_tas_2000.nc'

os.mkdir(temporary_destination_path) #Creating temporary directory to save results to (will delete this dir later)

land_only_mask = np.load(f'{mask_path}metum011_grid_land_filter.npy')
broadcasted_mask = np.broadcast_to(land_only_mask[np.newaxis,np.newaxis,np.newaxis,:,:],[8,3,456,392,504])

grid_cube = iris.load(grid_cube_path)[0][0]
for coord in grid_cube.aux_coords:
    grid_cube.remove_coord(coord)
grid_cube.coord('grid_longitude').guess_bounds()
grid_cube.coord('grid_latitude').guess_bounds()
weights = iris.analysis.cartography.area_weights(grid_cube, normalize=False) # area values in km^2
weights_masked = ma.masked_where(land_only_mask==False, weights)
weights_normalised = weights/np.nansum(weights_masked) # normalising by total area of ice sheet

ensemble_stl_decomp = np.load(f'{input_path}ensemble_stl_decomp_{variable}.npy')# returns shape [8,3,456,392,504]
ensemble_stl_decomp = ma.masked_where(broadcasted_mask==False, ensemble_stl_decomp) # masks to ice sheet only

if variable == 'temperature':
    ensemble_ice_sheet_only_total = np.nansum(ensemble_stl_decomp*weights_normalised,axis=(3,4)) # multiplying by normalised grid-cell area weight
else:
    ensemble_ice_sheet_only_total = np.nansum(ensemble_stl_decomp*weights,axis=(3,4)) # multiplying by grid-cell area weight

np.save(f'{temporary_destination_path}/ensemble_stl_decomp_{variable}_icesheetonly_total',ensemble_ice_sheet_only_total.data)

for file_name in os.listdir(temporary_destination_path):
    shutil.move(f'{temporary_destination_path}/{file_name}', f'{final_destination_path}{file_name}')    
    
os.rmdir(temporary_destination_path) 
