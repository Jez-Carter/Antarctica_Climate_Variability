# Loader Functions

import os
import numpy as np
import iris
from iris.coords import AuxCoord
from tqdm import tqdm

from src.helper_functions import apply_common_time_coordinates
from src.helper_functions import add_doy_month_year
from src.helper_functions import add_2d_latlon_aux_coords
from src.helper_functions import regrid
from src.helper_functions import remove_auxcoords
from src.helper_functions import concatenate_cubes

def retrieve_rawdata_filepath(data_basepath,RCM,resolution,driving_data,year,variable):
    if RCM == 'MetUM':
        if resolution == '044':
            if variable == 'mean_prsn' or variable == 'mean_snm':
                filepath = f'MetUM/044_6hourly/Antarctic_CORDEX_MetUM_0p44deg_6-hourly_{variable}_{year}.nc'   
            else:
                filepath = f'MetUM/044_3hourly/Antarctic_CORDEX_MetUM_0p44deg_3-hourly_{variable}_{year}.nc'       
        elif resolution == '011':
            if variable == 'mean_prsn' or variable == 'mean_snm':
                filepath = f'MetUM/011_6hourly/Antarctic_CORDEX_MetUM_0p11deg_6-hourly_{variable}_{year}.nc'   
            else:
                filepath = f'MetUM/011_3hourly/Antarctic_CORDEX_MetUM_0p11deg_3-hourly_{variable}_{year}.nc'    
    elif RCM == 'MAR':
        if driving_data == 'ERA_INT':
            filepath = f'MAR/MAR_ERAINT/MAR3.11_ERAI_{variable}_{year}_3h.nc4'
        elif driving_data == 'ERA5':
            filepath = f'MAR/MAR_ERA5/MAR3.11_{variable}_{year}_3h.nc4'
    elif RCM == 'RACMO':
        if 1979 <= year <= 1980:
            decade = 1979
        elif 1981 <= year <= 1990:
            decade = 1981
        elif 1991 <= year <= 2000:
            decade = 1991
        elif 2001 <= year <= 2010:
            decade = 2001
        elif 2011 <= year <= 2018:
            decade = 2011
        if driving_data == 'ERA_INT':
            filepath = f'RACMO/RACMO_ERAINT/{variable}.KNMI-{decade}.ANT27.ERAINx_RACMO2.4.1.3H.nc'
        elif driving_data == 'ERA5':
            filepath = f'RACMO/RACMO_ERA5/{variable}.KNMI-{decade}.ANT27.ERA5-3H_RACMO2.3p2.3H.nc'
    elif RCM == 'ERA-Interim':
        if variable == '2mtemp':
            filepath = f'ERA_Interim/era_int_6hourly_{variable}_{year}.nc'
        else:
            filepath = f'ERA_Interim/era_int_3hourly_{variable}_{year}.nc'
    elif RCM == 'ERA5':       
        if variable == '2mtemp':
            filepath = f'ERA5/era5_hourly_{variable}_{year}.grib'
        else:
            filepath = f'ERA5/era5_{variable}_{year}.grib'      
    return(data_basepath+filepath)
        
def load_raw_data(data_basepath,RCM,resolution,driving_data,year,variable):
    filepath = retrieve_rawdata_filepath(data_basepath,RCM,resolution,driving_data,year,variable)
    cubelist = iris.load(filepath)
    if RCM == 'MetUM':
        cube = cubelist[0]
        if variable == 'mean_prsn' or variable == 'mean_snm':
            cube.data = cube.data*21600 #converts /s average to 6hour accumulation
    elif RCM == 'MAR':
        cube = cubelist[0]
        if variable == 'TT':
            cube.data += 273.15
            cube.units = 'kelvin'
    elif RCM == 'RACMO':  
        if variable == 'snowfall':
            cube_long_name = 'Solid Precipitative Flux'
        elif variable == 'snowmelt':
            cube_long_name = 'Snow Melt Flux'
        elif variable == 't2m':
            cube_long_name = '2-m Temperature'
        for acube in cubelist:
            if acube.long_name==cube_long_name:
                cube = acube    
        if 2011 <= year <= 2018 and variable in ['precip','snowfall','snowmelt']:
            cube.data = cube.data*10800 #(per second to 3hourly accumulation)
        elif driving_data == 'ERA5' and variable in ['precip','snowfall','snowmelt']:
            cube.data = cube.data*10800 #(per second to 3hourly accumulation)    
    elif RCM =='ERA-Interim':         
        cube = cubelist[0]
        if variable in ('total_precip', 'snowfall','snowmelt'):
            cube.data = cube.data*10**3 # Going from units of tonnes to kilograms
            cube = cube[3::4,:,:] # Only keeping 12 hour accumulation measurements.    
    elif RCM =='ERA5':
        cube = cubelist[0]
        if variable in ('total_precip', 'snowfall','snowmelt'):
            cube.data = cube.data*10**3 # Going from units of tonnes to kilograms
    
    aux_coord_names = []
    for coord in cube.aux_coords:
        aux_coord_names.append(coord.standard_name)
        if coord.standard_name in ['height','forecast_period','forecast_reference_time','originating_centre']:
            cube.remove_coord(coord.standard_name)  
    for coord in cube.dim_coords:
        if coord.standard_name in ['height']:#,'forecast_period','forecast_reference_time','originating_centre']:
            cube = cube[tuple([slice(None) if cube.coord_dims(coord.standard_name)[0]!=dim else 0 for dim in range(cube.ndim)])]
            #complicated way of removing single valued dimension coordinates from cube.
            cube.remove_coord(coord.standard_name)
        
    if 'latitude' not in aux_coord_names:
        if RCM == 'MAR':
            MAR_mask_filepath = f'{data_basepath}/MAR/mar_land_sea_mask.nc' # where I get lat/lon coordinates from
            mar_mask_cube = iris.load(MAR_mask_filepath)[0]
            lons,lats = mar_mask_cube.coord('longitude').points,mar_mask_cube.coord('latitude').points
            cube.add_aux_coord(AuxCoord(points=lats, standard_name='latitude', units='degrees'),(1,2))
            cube.add_aux_coord(AuxCoord(points=lons, standard_name='longitude', units='degrees'),(1,2))
        else:
            add_2d_latlon_aux_coords(cube)              
    cube.attributes = None 
    return(cube)

def save_monthly_aggregated_data(data_basepath,RCM,resolution,driving_data,start_year,end_year,variable,aggregation,destination_path):
    for year in tqdm(np.arange(start_year,end_year+1,1)):
        cube = load_raw_data(data_basepath,RCM,resolution,driving_data,year,variable)
        apply_common_time_coordinates(cube)
        add_doy_month_year(cube)
        year_constraint1 = iris.Constraint(time=lambda cell: cell.point.year == year) 
        # This is done to filter out values for example 2001-01-01 00:00:00 when loading the year 2000 datafile
        year_constraint_2 = iris.Constraint(coord_values={'year':lambda cell: cell == year}) 
        # This is done to avoid issues with year-01-01 00:00:00 appearing as Dec
        cube = cube.extract(year_constraint1 & year_constraint_2)
        if aggregation == 'sum':
            cube = cube.aggregated_by(['month','year'],iris.analysis.SUM)
        elif aggregation == 'mean':
            hour_constraint = iris.Constraint(time=lambda cell: cell.point.hour in [0,6,12,18])
            # This is done to give all cubes equal instantaneous measurements of temperature before aggregating. 
            cube = cube.extract(hour_constraint) 
            cube = cube.aggregated_by(['month','year'],iris.analysis.MEAN)
        
        iris.save(cube, f'{destination_path}{RCM}_{resolution}_{driving_data}_{variable}_{year}.nc') 

def save_all_years_regrid(input_path,filename_no_year,start_year,end_year,grid_cube,method,destination_path):
        
    for year in tqdm(np.arange(start_year,end_year+1,1)):
        cube = iris.load(f'{input_path}{filename_no_year}_{year}.nc')[0]
        cubelist = []
        for i in np.arange(0,cube.shape[0],1): #Regridding each time-coordinate in turn. 
            cube_regrid = regrid(cube[i],grid_cube,method)
            cube_regrid.add_aux_coord(cube[i].coord('time'))
            cubelist.append(cube_regrid)   
        cubelist = iris.cube.CubeList(cubelist)
        cube = cubelist.merge_cube()    
        iris.save(cube, f'{destination_path}{filename_no_year}_{year}.nc')
        print(f'{filename_no_year}_{year}.nc')

def save_ensemble_numpy_array(input_path,filename_list_no_year,start_year,end_year,destination_full_path):

    ensemble_data = []
    years = np.arange(start_year,end_year+1,1)

    for filename in tqdm(filename_list_no_year):
        model_data = []
        for year in years:
            cube = iris.load(f'{input_path}{filename}_{year}.nc')[0]
            model_data.append(cube.data)  
        ensemble_data.append(np.concatenate(model_data,axis=0))

    np.save(destination_full_path,np.array(ensemble_data))

        

