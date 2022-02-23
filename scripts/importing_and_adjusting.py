#!/bin/env python

# This script loads the raw data and adjusts it so that the time coordinates are consistent between the datasets and so all the datasets contain 2D lat,lon coordinates. The data is also aggregated to monthly values and then saved, ready for regridding.

# Note runtime per year are approximately:
# MetUM(044):10s/year
# MetUM(011):70s/year
# MAR(ERAI):10s/year
# MAR(ERA5):10s/year
# RACMO(ERAI):50s/year
# RACMO(ERA5):70s/year
# ERAI:10s/year
# ERA5:50s/year
# So for ~40 years of data, for every model and 3 variables, total runtime ~ 9.5 hours

import timeit
import shutil
import os
from src.loader_functions import save_monthly_aggregated_data
import sys

variable = sys.argv[1]

start_year = 1981
end_year = 2018
input_path = '/gws/nopw/j04/lancs_atmos/users/carter10/Antarctica_Climate_Variability/Raw_Data/'
temporary_destination_path = f'/work/scratch-pw/carter10/import_and_adjust_{variable}'
final_destination_path = '/gws/nopw/j04/lancs_atmos/users/carter10/Antarctica_Climate_Variability/Postprocessed_Data'

os.mkdir(temporary_destination_path) #Creating temporary directory to save results to (will delete this dir later)

time = timeit.default_timer()

if variable == 'temperature':
    variable_names = ['tas','TT','t2m','2mtemp']
    aggregation = 'mean'
elif variable == 'snowfall'
    variable_names = ['mean_prsn','SNF','snowfall','snowfall']
    aggregation = 'sum'
elif variable == 'melt'
    variable_names = ['mean_snm','ME','snowmelt','snowmelt']
    aggregation = 'sum'

save_monthly_aggregated_data(input_path,'MetUM','044','ERA_INT',start_year,end_year,variable_names[0],aggregation,f'{temporary_destination_path}/')
save_monthly_aggregated_data(input_path,'MetUM','011','ERA_INT',start_year,end_year,variable_names[0],aggregation,f'{temporary_destination_path}/')
save_monthly_aggregated_data(input_path,'MAR','35','ERA_INT',start_year,end_year,variable_names[1],aggregation,f'{temporary_destination_path}/')
save_monthly_aggregated_data(input_path,'MAR','35','ERA5',start_year,end_year,variable_names[1],aggregation,f'{temporary_destination_path}/')
save_monthly_aggregated_data(input_path,'RACMO','27','ERA_INT',start_year,end_year,variable_names[2],aggregation,f'{temporary_destination_path}/')
save_monthly_aggregated_data(input_path,'RACMO','27','ERA5',start_year,end_year,variable_names[2],aggregation,f'{temporary_destination_path}/')
save_monthly_aggregated_data(input_path,'ERA-Interim','79','NA',start_year,end_year,variable_names[3],aggregation,f'{temporary_destination_path}/')
save_monthly_aggregated_data(input_path,'ERA5','31','NA',start_year,end_year,variable_names[3],aggregation,f'{temporary_destination_path}/')

for file_name in os.listdir(temporary_destination_path):
    shutil.move(f'{temporary_destination_path}/{file_name}', f'{final_destination_path}{file_name}')

os.rmdir(temporary_destination_path)        
print(f'Total Time Taken {variable}:', timeit.default_timer() - time)
