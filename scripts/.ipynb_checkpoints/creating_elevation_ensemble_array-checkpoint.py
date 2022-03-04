#!/bin/env python

# This script returns an ensemble elevation array, with the elevations from each model regridded onto a common MetUM(011) grid. Shape [6,392,504]

import os 
import shutil
import numpy as np
import iris
from src.helper_functions import add_2d_latlon_aux_coords
from src.helper_functions import regrid
from iris.coords import AuxCoord

input_path = '/gws/nopw/j04/lancs_atmos/users/carter10/Antarctica_Climate_Variability/Raw_Data/'
temporary_destination_path = f'/work/scratch-pw/carter10/regridding_elevation_data'
final_destination_path = '/gws/nopw/j04/lancs_atmos/users/carter10/Antarctica_Climate_Variability/Ensemble_Array_Data/'

os.mkdir(temporary_destination_path) #Creating temporary directory to save results to (will delete this dir later)


models = ['ERAI','ERA5','MetUM(044)','MetUM(011)','MAR','RACMO'] 
filepaths = [f'{input_path}ERA_Interim/ERA_Interim_Geopotential.nc',
             f'{input_path}ERA5/ERA5_Geopotential.grib',
             f'{input_path}MetUM/044_fixed/Antarctic_CORDEX_MetUM_0p44deg_orog.nc',
             f'{input_path}MetUM/011_fixed/Antarctic_CORDEX_MetUM_0p11deg_orog.nc',
             f'{input_path}MAR/MARcst-AN35km-176x148.cdf',
             f'{input_path}RACMO/Height_latlon_ANT27.nc']

regridded_elevation_data=[]
for model,filepath in zip(models,filepaths):
    cubelist = iris.load(filepath)
    if model in ['MAR','RACMO']:    
        for acube in cubelist:
            if acube.long_name=='Surface height' or acube.long_name=='height above the surface':
                cube = acube 
    else:
        cube = cubelist[0]
    
    if model == 'RACMO':
        ref_cube_filepath = '/gws/nopw/j04/lancs_atmos/users/carter10/Antarctica_Climate_Variability/Postprocessed_Data/RACMO_27_ERA_INT_snowfall_2000.nc'
        ref_cube = iris.load(ref_cube_filepath)[0].collapsed('time',iris.analysis.SUM)
        grid_latitude = ref_cube.coord('grid_latitude')
        grid_longitude = ref_cube.coord('grid_longitude')
        cube = iris.cube.Cube(
            cube.data,
            long_name='Elevation',
            dim_coords_and_dims=[(grid_latitude,0),(grid_longitude,1)])
    
    if model in ['ERAI','ERA5']:
        cube.data = cube.data/9.80665
    
    aux_coord_names = []
    for coord in cube.aux_coords:
        aux_coord_names.append(coord.standard_name)
    if 'latitude' not in aux_coord_names:
        if model == 'MAR':
            MAR_mask_filepath = f'{input_path}/MAR/mar_land_sea_mask.nc' # where I get lat/lon coordinates from
            mar_mask_cube = iris.load(MAR_mask_filepath)[0]
            lons,lats = mar_mask_cube.coord('longitude').points,mar_mask_cube.coord('latitude').points
            cube.add_aux_coord(AuxCoord(points=lats, standard_name='latitude', units='degrees'),(0,1))
            cube.add_aux_coord(AuxCoord(points=lons, standard_name='longitude', units='degrees'),(0,1))
        else:
            add_2d_latlon_aux_coords(cube)  
    
    grid_cube_filepath = '/gws/nopw/j04/lancs_atmos/users/carter10/Antarctica_Climate_Variability/Postprocessed_Data/MetUM_011_ERA_INT_tas_2000.nc'
    grid_cube = iris.load(grid_cube_filepath)[0][0,:,:]
    method='cubic'
    cube = regrid(cube,grid_cube,method)
    regridded_elevation_data.append(cube.data)
    
np.save(f'{temporary_destination_path}/ensemble_elevations',np.array(regridded_elevation_data))

for file_name in os.listdir(temporary_destination_path):
    shutil.move(f'{temporary_destination_path}/{file_name}', f'{final_destination_path}{file_name}')    
    
os.rmdir(temporary_destination_path) 
