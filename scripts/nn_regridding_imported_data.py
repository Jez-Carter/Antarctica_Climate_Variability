#!/bin/env python

# This script regrids the imported data onto common grids and saves the output. 

import timeit
import shutil
import os
import iris
from src.loader_functions import save_all_years_regrid
import sys

variable = sys.argv[1]

temperature_filename_list_no_year = ['ERA-Interim_79_NA_2mtemp',
           'ERA5_31_NA_2mtemp', 
           'MetUM_044_ERA_INT_tas',
           'MetUM_011_ERA_INT_tas',
           'MAR_35_ERA_INT_TT',
           'MAR_35_ERA5_TT',
           'RACMO_27_ERA_INT_t2m',
           'RACMO_27_ERA5_t2m']

snowfall_filename_list_no_year = ['ERA-Interim_79_NA_snowfall',
           'ERA5_31_NA_snowfall',
           'MetUM_044_ERA_INT_mean_prsn',
           'MetUM_011_ERA_INT_mean_prsn',
           'MAR_35_ERA_INT_SNF',
           'MAR_35_ERA5_SNF',
           'RACMO_27_ERA_INT_snowfall',
           'RACMO_27_ERA5_snowfall']

melt_filename_list_no_year = ['ERA-Interim_79_NA_snowmelt',
           'ERA5_31_NA_snowmelt', 
           'MetUM_044_ERA_INT_mean_snm',
           'MetUM_011_ERA_INT_mean_snm',
           'MAR_35_ERA_INT_ME',
           'MAR_35_ERA5_ME',
           'RACMO_27_ERA_INT_snowmelt',
           'RACMO_27_ERA5_snowmelt']

input_path = '/gws/nopw/j04/lancs_atmos/users/carter10/Antarctica_Climate_Variability/Postprocessed_Data/'
temporary_destination_path = f'/work/scratch-pw/carter10/regridding_imported_data_{variable}'
final_destination_path = '/gws/nopw/j04/lancs_atmos/users/carter10/Antarctica_Climate_Variability/NN_Regridded_Data/'

os.mkdir(temporary_destination_path) #Creating temporary directory to save results to (will delete this dir later)

start_year=1981
end_year=2018
grid_cube = iris.load(input_path+'MetUM_011_ERA_INT_tas_2000.nc')[0][0,:,:]
method='nearest'

if variable == 'temperature':
    filename_list_no_year = temperature_filename_list_no_year
elif variable == 'snowfall':
    filename_list_no_year = snowfall_filename_list_no_year
elif variable == 'melt':
    filename_list_no_year = melt_filename_list_no_year

time = timeit.default_timer()
for filename_no_year in filename_list_no_year:
    save_all_years_regrid(input_path,filename_no_year,start_year,end_year,grid_cube,method,f'{temporary_destination_path}/')

for file_name in os.listdir(temporary_destination_path):
    shutil.move(f'{temporary_destination_path}/{file_name}', f'{final_destination_path}{file_name}')

os.rmdir(temporary_destination_path) 
    
print(f'Total Time Taken {variable}:', timeit.default_timer() - time)
