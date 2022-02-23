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
            cube.data = cube.data*10**3 # Going from units of grams to kilograms
            cube = cube[3::4,:,:] # Only keeping 12 hour accumulation measurements.    
    elif RCM =='ERA5':
        cube = cubelist[0]
        if variable in ('total_precip', 'snowfall','snowmelt'):
            cube.data = cube.data*10**3 # Going from units of grams to kilograms
    
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

        
# def memory_efficient_regridding(input_path,cube_filename_no_year,start_year,end_year,regridding_cube_filename):
#     #RCM,resolution,driving_data,start_year,end_year,regridding_cube):
    
#     regridding_cube = iris.load(input_path+regridding_cube_filename)[0]
    
#     for i in np.arange(start_year,end_year+1,1):
        
#         #cube = iris.load(RCM+'_'+resolution+'_'+driving_data+'_'+str(i)+'.nc')[0]
        
#         cube_filename = cube_filename_no_year+'_'+str(i)+'.nc'
        
#         cube = iris.load(input_path + cube_filename)[0]
        
#         cube_cg = regrid_onto_regriddingcube(cube,regridding_cube)
        
#         iris.save(cube_cg, input_path + cube_filename_no_year + '_cg_' + str(i) + '.nc')
#         #iris.save(cube_cg, RCM+'_'+resolution+'_'+driving_data+'_'+str(i)+'_'+'cg'+'.nc')

# def examine_raw_data(RCM,resolution,driving_data,start_year,end_year,variable,cube_or_cubelist):
    
#     os.chdir(os.path.expanduser("~"))
#     files = []
    
#     if RCM == 'MetUM':
#         if resolution == '044':
#             os.chdir('Shared_Storage/Google_Bucket_Transfer/CORDEX_Data/MetUM/044_3hourly/')
            
#             for i in np.arange(start_year,end_year+1,1):
#                 file = 'Antarctic_CORDEX_MetUM_0p44deg_3-hourly_tas_'+str(i)+'.nc'
#                 files.append(file)
            
#             cubelist = iris.load(files)
            
#         elif resolution == '011':
#             os.chdir('Shared_Storage/Google_Bucket_Transfer/CORDEX_Data/MetUM/011_3hourly/')
            
#             for i in np.arange(start_year,end_year+1,1):
#                 file = 'Antarctic_CORDEX_MetUM_0p11deg_3-hourly_tas_'+str(i)+'.nc'
#                 files.append(file)
            
#             cubelist = iris.load(files)
        
#         cube = cubelist.concatenate()[0]
    
#     elif RCM == 'MAR':
#         os.chdir('Shared_Storage/Google_Bucket_Transfer/CORDEX_Data/MAR/')
        
#         if driving_data == 'ERA_INT':
#             os.chdir('MAR_CORDEX_Daily_ERAINT_Data/')
            
#             for i in os.listdir():
#                 filename_start_year = int(i.split('daily-TT-MAR_ERAint-')[1].split('.nc')[0].split('-')[0])
#                 filename_end_year = int(i.split('daily-TT-MAR_ERAint-')[1].split('.nc')[0].split('-')[1])

#                 if filename_start_year <= start_year <= filename_end_year:
#                     files.append(i)
#                 elif filename_start_year <= end_year <= filename_end_year:
#                     files.append(i)
                
#             cubelist = iris.load(files)
#             cube = concatenate_cubes(cubelist)
#             years_constraint = iris.Constraint(time=lambda cell: start_year <= cell.point.year <= end_year)
#             cube = cube.extract(years_constraint)
                    
#         elif driving_data == 'ERA5':
#             os.chdir('MAR_CORDEX_3hourly_ERA5/')
            
#             for i in np.arange(start_year,end_year+1,1):
#                 file = 'MAR3.11_TT_'+str(i)+'_3h.nc4'
#                 files.append(file)
            
#             cubelist = iris.load(files)
#             cube = concatenate_cubes(cubelist)
        
#     elif RCM == 'RACMO':
#         os.chdir('Shared_Storage/Google_Bucket_Transfer/CORDEX_Data/RACMO/')
#         files = []
        
#         if driving_data == 'ERA_INT':
#             os.chdir('3Hourly/')

#             example_filename = 'snowfall.KNMI-1979.ANT27.ERAINx_RACMO2.4.1.3H.nc'
#             year_left_split = f'{variable}.KNMI-'
#             year_right_split = '.ANT27.ERAINx_RACMO2.4.1.3H.nc'

#             for i in os.listdir():
                
#                 if i.split('.')[0] == variable:
    
#                     filename_start_year = int(i.split(year_left_split)[1].split(year_right_split)[0])
        
#                     if filename_start_year == 1979:
#                         filename_end_year = 1980
#                     else:
#                         filename_end_year = filename_start_year+9
                        
#                     if filename_start_year <= start_year <= filename_end_year:
#                         files.append(i)
#                     elif filename_start_year <= end_year <= filename_end_year:
#                         files.append(i)
            
#             cubelist = iris.load(files)
            
#             if variable == 'snowmelt':
#                 index = 6
#             else:
#                 index = 7
            
#             print(index)
#             #cube = cubelist.concatenate()[index][:,0,:,:]
#             #years_constraint = iris.Constraint(time=lambda cell: start_year <= cell.point.year <= end_year)
#             #cube = cube.extract(years_constraint)
           
#     elif RCM =='ERA-Interim':
#         os.chdir('Shared_Storage/Google_Bucket_Transfer/Reanalysis_Data/ERA_Interim/')
#         for i in np.arange(start_year,end_year+1,1):
#             file = 'era_int_6hourly_2mtemp_'+str(i)+'.nc'
#             files.append(file)
            
#         cubelist = iris.load(files)
#         cube = cubelist.concatenate()[0]
        
#     elif RCM =='ERA5':
#         os.chdir('Shared_Storage/Google_Bucket_Transfer/Reanalysis_Data/ERA5/')
#         for i in np.arange(start_year,end_year+1,1):
#             file = 'era5_hourly_2mtemp_'+str(i)+'.grib'
#             files.append(file)
            
#         cubelist = iris.load(files)
#         cube = cubelist.concatenate()[0]
        
#     if cube_or_cubelist == 'cube':
#         return(cube)
    
#     elif cube_or_cubelist == 'cubelist':
#         return(cubelist)

# def memory_efficient_loading(RCM,resolution,driving_data,start_year,end_year,variable):
#     for i in tqdm(np.arange(start_year,end_year+1,1)):
#         cube = load_raw_data(RCM,resolution,driving_data,i,i,variable)
        
#         os.chdir(os.path.expanduser("~"))
#         os.chdir('Shared_Storage/Google_Bucket_Transfer/Postprocessed_Data')
#         iris.save(cube, RCM+'_'+resolution+'_'+driving_data+'_'+variable+'_'+str(i)+'.nc')
        
# def save_all_years_regrid(filename_no_year,start_year,end_year,aggregation,grid_cube,method):
        
#     for i in tqdm(np.arange(start_year,end_year+1,1)):
#         os.chdir(os.path.expanduser("~"))
#         os.chdir('Shared_Storage/Google_Bucket_Transfer/Postprocessed_Data')
#         cube = iris.load(filename_no_year+'_'+str(i)+'.nc')[0]
#         if filename_no_year == 'RACMO_27_ERA5_precip':
#             cube = cube[:,0,:,:]
#             cube.data = cube.data*10800
#         elif filename_no_year == 'RACMO_27_ERA5_snowfall':
#             cube = cube[:,0,:,:]
#             cube.data = cube.data*10800
#         elif filename_no_year == 'RACMO_27_ERA5_snowmelt':
#             cube = cube[:,0,:,:]
#             cube.data = cube.data*10800
            
#         if aggregation == 'sum':
#             cube = cube.collapsed(['year'],iris.analysis.SUM)
#         elif aggregation == 'mean':
#             cube = cube.collapsed(['year'],iris.analysis.MEAN)

#         cube = regrid(cube,grid_cube,method)
#         os.chdir(os.path.expanduser("~"))
#         os.chdir('Shared_Storage/Google_Bucket_Transfer/MetUMGrid_Data/Yearly')
#         iris.save(cube, filename_no_year+'_'+str(i)+'.nc')
        
        
# def memory_efficient_common_mask_common_time_cubes(filename_list_no_year,start_year,end_year):
#     os.chdir(os.path.expanduser("~"))
#     os.chdir('Shared_Storage/Google_Bucket_Transfer/Postprocessed_Data')

#     for i in tqdm(np.arange(start_year,end_year+1,1)):
        
#         list_of_cubes = []
        
#         for name in filename_list_no_year:
#             cube = iris.load(name+'_'+str(i)+'.nc')[0]
#             list_of_cubes.append(cube)
        
#         common_domain_list_of_cubes = apply_common_mask(list_of_cubes)
        
#         common_domain_common_time_list_of_cubes = apply_common_time_coord(common_domain_list_of_cubes)
        
#         #common_domain_list_of_cubes_daily = []
        
#         #for cd_cube in common_domain_list_of_cubes:
#         #    daily_cube = aggregate_to_daily(cd_cube)
#         #    common_domain_list_of_cubes_daily.append(daily_cube)
                    
#         for j in np.arange(0,len(common_domain_common_time_list_of_cubes),1):   
#             #print(filename_list_no_year[j]+'_cm_ct_'+str(i)+'.nc')
#             iris.save(common_domain_common_time_list_of_cubes[j], filename_list_no_year[j]+'_cm_ct_'+str(i)+'.nc')        

# def memory_efficient_relative_cubes(RCM_list,start_year,end_year):
#     os.chdir(os.path.expanduser("~"))
#     os.chdir('Shared_Storage/Google_Bucket_Transfer/Postprocessed_Data')
    
#     for i in tqdm(np.arange(start_year,end_year+1,1)):
#         list_of_cubes = []
        
#         for name in RCM_list:
#             cube = iris.load(name+'_'+str(i)+'_'+'cg'+'_'+'cm'+'_'+'daily'+'.nc')[0]
#             list_of_cubes.append(cube)
        
#         relative_cubes = calculate_relative_cubes(list_of_cubes)
        
#         for j in np.arange(0,len(relative_cubes),1):  
#             iris.save(relative_cubes[j], RCM_list[j]+'_'+str(i)+'_'+'cg'+'_'+'cm'+'_'+'daily'+'_'+'rel'+'.nc')        
                              
# def memory_efficient_stddev_cubes(RCM_list,start_year,end_year):
#     os.chdir(os.path.expanduser("~"))
#     os.chdir('Shared_Storage/Google_Bucket_Transfer/Postprocessed_Data')
    
#     for i in tqdm(np.arange(start_year,end_year+1,1)):
        
#         list_of_cubes=[]
        
#         for name in RCM_list:
#             cube = iris.load(name+'_'+str(i)+'_'+'cg'+'_'+'cm'+'_'+'daily'+'.nc')[0]
#             list_of_cubes.append(cube)
            
#         stddev_cube = calculate_ensemble_stddev(list_of_cubes)
        
#         iris.save(stddev_cube,'stddev'+'_'+str(i)+'.nc')
        
# def memory_efficient_mean_cubes(RCM_list,start_year,end_year):
#     os.chdir(os.path.expanduser("~"))
#     os.chdir('Shared_Storage/Google_Bucket_Transfer/Postprocessed_Data')
    
#     for i in tqdm(np.arange(start_year,end_year+1,1)):
        
#         list_of_cubes=[]
        
#         for name in RCM_list:
#             cube = iris.load(name+'_'+str(i)+'_'+'cg'+'_'+'cm'+'_'+'daily'+'.nc')[0]
#             list_of_cubes.append(cube)
            
#         mean_cube = calculate_ensemble_mean(list_of_cubes)
        
#         iris.save(mean_cube,'mean'+'_'+str(i)+'.nc')
        
# def memory_efficient_temporal_mean_cubes(RCM_list,start_year,end_year):
#     os.chdir(os.path.expanduser("~"))
#     os.chdir('Shared_Storage/Google_Bucket_Transfer/Postprocessed_Data')
        
#     for i in tqdm(np.arange(start_year,end_year+1,1)):
        
#         list_of_cubes = []
        
#         for name in RCM_list:
#             cube = iris.load(name+'_'+str(i)+'_'+'cg'+'_'+'cm'+'_'+'daily'+'.nc')[0]
#             list_of_cubes.append(cube)
            
#         temporally_averaged_list_of_cubes = []
        
#         for cube in list_of_cubes:
#             temporally_averaged_cube = cube.collapsed(['time'],iris.analysis.MEAN)
#             temporally_averaged_list_of_cubes.append(temporally_averaged_cube)
                    
#         for j in np.arange(0,len(temporally_averaged_list_of_cubes),1):  
#             iris.save(temporally_averaged_list_of_cubes[j], RCM_list[j]+'_'+str(i)+'_'+'cg'+'_'+'cm'+'_'+'temporal'+'_'+'avg'+'.nc')            
                      
# def memory_efficient_temporal_stddev_cubes(RCM_list,start_year,end_year):
#     os.chdir(os.path.expanduser("~"))
#     os.chdir('Shared_Storage/Google_Bucket_Transfer/Postprocessed_Data')
    
#     for name in RCM_list:
        
#         list_of_cubes = []
        
#         for i in tqdm(np.arange(start_year,end_year+1,1)):
#             cube = iris.load(name+'_'+str(i)+'_'+'cg'+'_'+'cm'+'_'+'daily'+'.nc')[0]
#             list_of_cubes.append(cube)
            
#         cubelist = iris.cube.CubeList(list_of_cubes)
#         all_years_cube = cubelist.concatenate_cube()
#         stddev_cube = all_years_cube.collapsed('time',iris.analysis.STD_DEV)

#         iris.save(stddev_cube, name+'_'+'cg'+'_'+'cm'+'_'+'temporal'+'_'+'stddev'+'.nc')

# def memory_efficient_relative_temporal_mean_cubes(RCM_list,start_year,end_year):
    
#     os.chdir(os.path.expanduser("~"))
#     os.chdir('Shared_Storage/Google_Bucket_Transfer/Postprocessed_Data')
    
#     for i in tqdm(np.arange(start_year,end_year+1,1)):
        
#         list_of_cubes = []

#         for name in RCM_list:
#             cube = iris.load(name+'_'+str(i)+'_'+'cg'+'_'+'cm'+'_'+'daily'+'_'+'relative'+'.nc')[0]
#             list_of_cubes.append(cube)

#         temporally_averaged_list_of_cubes = []

#         for cube in list_of_cubes:
#             temporally_averaged_cube = cube.collapsed(['time'],iris.analysis.MEAN)
#             temporally_averaged_list_of_cubes.append(temporally_averaged_cube)

#         for j in np.arange(0,len(temporally_averaged_list_of_cubes),1):  
#             iris.save(temporally_averaged_list_of_cubes[j], RCM_list[j]+'_'+str(i)+'_'+'cg'+'cm'+'_'+'relative'+'_'+'temporal'+'_'+'avg'+'.nc')
            
# def memory_efficient_relative_temporal_stddev_cubes(RCM_list,start_year,end_year):
    
#     os.chdir(os.path.expanduser("~"))
#     os.chdir('Shared_Storage/Google_Bucket_Transfer/Postprocessed_Data')
    
#     for name in RCM_list:
        
#         all_years_list = []
        
#         for i in tqdm(np.arange(start_year,end_year+1,1)):
#             cube = iris.load(name+'_'+str(i)+'_'+'cg'+'_'+'cm'+'_'+'daily'+'_'+'rel'+'.nc')[0]
#             all_years_list.append(cube)
            
#         all_years_cubelist = iris.cube.CubeList(all_years_list)
#         all_years_cube = all_years_cubelist.concatenate_cube()
        
#         temporal_stddev_cube = all_years_cube.collapsed(['time'],iris.analysis.STD_DEV)
        
#         iris.save(temporal_stddev_cube, name+'_'+'cg'+'cm'+'_'+'rel'+'_'+'temporal'+'_'+'stddev'+'.nc')
        
# def memory_efficient_common_mask_daily_timeseries_cubes(RCM_list,start_year,end_year):
#     os.chdir(os.path.expanduser("~"))
#     os.chdir('Shared_Storage/Google_Bucket_Transfer/Postprocessed_Data')
        
#     for i in tqdm(np.arange(start_year,end_year+1,1)):
        
#         list_of_cubes = []
        
#         for name in RCM_list:
#             cube = iris.load(name+'_'+str(i)+'_'+'cg'+'_'+'cm'+'_'+'daily'+'.nc')[0]
#             timeseries_cube = cube.collapsed(['grid_latitude','grid_longitude'],iris.analysis.MEAN)
#             iris.save(timeseries_cube, name+'_'+str(i)+'_'+'cg'+'_'+'cm'+'_'+'daily'+'_'+'timeseries'+'.nc')
        
# def memory_efficient_common_mask_daily_timeseries_land_cubes(RCM_list,start_year,end_year):
    
#     os.chdir(os.path.expanduser("~"))
#     os.chdir('Shared_Storage/Google_Bucket_Transfer/Postprocessed_Data')
        
#     for i in tqdm(np.arange(start_year,end_year+1,1)):
        
#         list_of_cubes = []
        
#         for name in RCM_list:
#             cube = iris.load(name+'_'+str(i)+'_'+'cg'+'_'+'cm'+'_'+'daily'+'.nc')[0]
#             cube = mask_to_land_only(cube,cube[0])
#             os.chdir(os.path.expanduser("~"))
#             os.chdir('Shared_Storage/Google_Bucket_Transfer/Postprocessed_Data')
#             timeseries_cube = cube.collapsed(['grid_latitude','grid_longitude'],iris.analysis.MEAN)
#             iris.save(timeseries_cube, name+'_'+str(i)+'_'+'cg'+'_'+'cm'+'_'+'daily'+'_'+'timeseries'+'_'+'land'+'.nc')
            
# def memory_efficient_relative_daily_timeseries_cubes(RCM_list,start_year,end_year):
    
#     os.chdir(os.path.expanduser("~"))
#     os.chdir('Shared_Storage/Google_Bucket_Transfer/Postprocessed_Data')
    
#     for i in tqdm(np.arange(start_year,end_year+1,1)):
        
#         list_of_cubes=[]
        
#         for name in RCM_list:
#             cube = iris.load(name+'_'+str(i)+'_'+'cg'+'_'+'cm'+'_'+'daily'+'_'+'timeseries'+'.nc')[0]
#             list_of_cubes.append(cube)
            
#         relative_list_of_cubes = calculate_relative_cubes(list_of_cubes)
        
#         for j in np.arange(0,len(relative_list_of_cubes),1):  
#             iris.save(relative_list_of_cubes[j], RCM_list[j]+'_'+str(i)+'_'+'cg'+'_'+'cm'+'_'+'daily'+'_'+'timeseries'+'_'+'relative'+'.nc')
            
# def memory_efficient_relative_daily_timeseries_land_cubes(RCM_list,start_year,end_year):
    
#     os.chdir(os.path.expanduser("~"))
#     os.chdir('Shared_Storage/Google_Bucket_Transfer/Postprocessed_Data')
    
#     for i in tqdm(np.arange(start_year,end_year+1,1)):
        
#         list_of_cubes=[]
        
#         for name in RCM_list:
#             cube = iris.load(name+'_'+str(i)+'_'+'cg'+'_'+'cm'+'_'+'daily'+'_'+'timeseries'+'_'+'land'+'.nc')[0]
#             list_of_cubes.append(cube)
            
#         relative_list_of_cubes = calculate_relative_cubes(list_of_cubes)
        
#         for j in np.arange(0,len(relative_list_of_cubes),1):  
#             iris.save(relative_list_of_cubes[j], RCM_list[j]+'_'+str(i)+'_'+'cg'+'_'+'cm'+'_'+'daily'+'_'+'timeseries'+'_'+'land'+'_'+'relative'+'.nc')

# def load_raw_data(input_path,RCM,resolution,driving_data,start_year,end_year,variable):
#     os.chdir(input_path)
#     files = []
    
#     if RCM == 'MetUM':
#         if resolution == '044':
#             os.chdir(input_path + 'CORDEX_Data/MetUM/')
#             if variable == 'mean_prsn' or variable == 'mean_snm':
#                 os.chdir('044_6hourly/')
#                 filename_start = 'Antarctic_CORDEX_MetUM_0p44deg_6-hourly_'
#             else:
#                 os.chdir('044_3hourly/')
#                 filename_start = 'Antarctic_CORDEX_MetUM_0p44deg_3-hourly_'
            
#         elif resolution == '011':
#             os.chdir(input_path + 'CORDEX_Data/MetUM/')
            
#             if variable == 'mean_prsn' or variable == 'mean_snm':
#                 os.chdir('011_6hourly/')
#                 filename_start = 'Antarctic_CORDEX_MetUM_0p11deg_6-hourly_'
#             else:
#                 os.chdir('011_3hourly/')
#                 filename_start = 'Antarctic_CORDEX_MetUM_0p11deg_3-hourly_'
            
#         for i in np.arange(start_year,end_year+1,1):
#             file = filename_start+variable+'_'+str(i)+'.nc'
#             files.append(file)
            
#         cubelist = iris.load(files)
#         cube = cubelist.concatenate()[0]
#         apply_common_time_coordinates(cube)
#         add_doy_month_year(cube)
#         add_2d_latlon_aux_coords(cube)
#         cube.coord('grid_longitude').guess_bounds()
#         cube.coord('grid_latitude').guess_bounds()
#         cube.attributes = {}
#         cube.remove_coord('forecast_period')
#         cube.remove_coord('forecast_reference_time')
        
#         if variable == 'tas':
#             cube.remove_coord('height')
            
#         #converting to 3hourly accumulation units 
#         if variable == 'mean_pr':
#             cube.data = cube.data*10800
#             cube.coord('time').points = cube.coord('time').points+0.0625 #Adding 1.5hours to each time coordinate
#             apply_common_time_coordinates(cube)
            
#         elif variable == 'mean_prsn' or variable == 'mean_snm':
#             cube.data = cube.data*21600
#             cube.coord('time').points = cube.coord('time').points+0.1250 #Adding 3hours to each time coordinate
#             apply_common_time_coordinates(cube)
 
#     elif RCM == 'MAR':
#         os.chdir(input_path + 'CORDEX_Data/MAR/')
        
#         mar_land_sea_mask = iris.load('mar_land_sea_mask.nc')[0]
        
#         if driving_data == 'ERA_INT':
#             os.chdir('MAR_CORDEX_3hourly_ERAINT/')
            
#             # example_filename = 'MAR3.11_ERAI_ME_1984_3h.nc4'
            
#             for i in np.arange(start_year,end_year+1,1):
#                 file = f'MAR3.11_ERAI_{variable}_'+str(i)+'_3h.nc4'
#                 files.append(file)
            
#             cubelist = iris.load(files)
#             cube = concatenate_cubes(cubelist)
                    
#         elif driving_data == 'ERA5':
#             os.chdir('MAR_CORDEX_3hourly_ERA5/')
            
#             # example_filename = 'MAR3.11_SNF_2009_3h.nc4'
            
#             for i in np.arange(start_year,end_year+1,1):
#                 file = f'MAR3.11_{variable}_'+str(i)+'_3h.nc4'
#                 files.append(file)
            
#             cubelist = iris.load(files)
#             cube = concatenate_cubes(cubelist)
                
#         years_constraint = iris.Constraint(time=lambda cell: start_year <= cell.point.year <= end_year)
#         cube = cube.extract(years_constraint)
#         add_doy_month_year(cube)
        
#         if variable == 'TT':
#             cube.data += 273.15
#             cube.units = 'kelvin'
#         apply_common_time_coordinates(cube)
        
#         lons,lats = mar_land_sea_mask.coord('longitude').points,mar_land_sea_mask.coord('latitude').points
        
#         cube.add_aux_coord(AuxCoord(points=lats, standard_name='latitude', units='degrees'),(1,2))
#         cube.add_aux_coord(AuxCoord(points=lons, standard_name='longitude', units='degrees'),(1,2))

#         cube.attributes = {}
        
#     elif RCM == 'RACMO':
#         os.chdir(input_path + 'CORDEX_Data/RACMO/')
#         files = []
        
#         if driving_data == 'ERA_INT':
#             os.chdir('3Hourly/')

#             # example_filename = 'snowfall.KNMI-1979.ANT27.ERAINx_RACMO2.4.1.3H.nc'
#             year_left_split = f'{variable}.KNMI-'
#             year_right_split = '.ANT27.ERAINx_RACMO2.4.1.3H.nc'
            
#         elif driving_data == 'ERA5':
#             os.chdir('RACMO_27_3Hourly_ERA5/')

#             # example_filename = 'snowfall.KNMI-1979.ANT27.ERAINx_RACMO2.4.1.3H.nc'
#             year_left_split = f'{variable}.KNMI-'
#             year_right_split = '.ANT27.ERA5-3H_RACMO2.3p2.3H.nc'

#         for i in os.listdir():

#             if i.split('.')[0] == variable:

#                 filename_start_year = int(i.split(year_left_split)[1].split(year_right_split)[0])

#                 if filename_start_year == 1979:
#                     filename_end_year = 1980
#                 else:
#                     filename_end_year = filename_start_year+9

#                 if filename_start_year <= start_year <= filename_end_year:
#                     files.append(i)
#                 elif filename_start_year <= end_year <= filename_end_year:
#                     files.append(i)
        
#         if variable == 'snowfall':
#             cube_long_name = 'Solid Precipitative Flux'
#         elif variable == 'snowmelt':
#             cube_long_name = 'Snow Melt Flux'
#         elif variable == 't2m':
#             cube_long_name = '2-m Temperature'
            
#         list_of_cubes = []
#         for file in files:
#             cubelist = iris.load(file)
#             for a_cube in cubelist:
#                 if a_cube.long_name==cube_long_name:
#                     cube = a_cube    
#                     cube.coord('time').attributes = None

#                     if file in ('precip.KNMI-2011.ANT27.ERAINx_RACMO2.4.1.3H.nc','snowfall.KNMI-2011.ANT27.ERAINx_RACMO2.4.1.3H.nc','snowmelt.KNMI-2011.ANT27.ERAINx_RACMO2.4.1.3H.nc'):
#                         cube.data = cube.data*10800 #(seconds to 3hourly precip)

#                     elif driving_data == 'ERA5' and (variable == 'snowmelt' or variable == 'snowfall'):
#                         cube.data = cube.data*10800 #(seconds to 3hourly precip)
#                     list_of_cubes.append(cube)

#         cubelist = iris.cube.CubeList(list_of_cubes)
#         cube = cubelist.concatenate_cube()[:,0,:,:]
                
#         years_constraint = iris.Constraint(time=lambda cell: start_year <= cell.point.year <= end_year)
#         cube = cube.extract(years_constraint)
        
#         cube.coord('time').attributes = {}
#         apply_common_time_coordinates(cube)
          
#         add_doy_month_year(cube)
#         cube.remove_coord('latitude')
#         cube.remove_coord('longitude')
#         add_2d_latlon_aux_coords(cube)
#         add_shifted_2d_latlon_aux_coords(cube)
#         cube.coord('grid_longitude').guess_bounds()
#         cube.coord('grid_latitude').guess_bounds()
#         cube.remove_coord('height')
#         cube.attributes = {}
           
#     elif RCM =='ERA-Interim':
        
#         #filename_example = 'era_int_6hourly_total_precip_2002.nc'
        
#         if variable == '2mtemp':
#             filename_start = 'era_int_6hourly_'
#         else:
#             filename_start = 'era_int_3hourly_'
        
#         os.chdir(input_path + 'Reanalysis_Data/ERA_Interim/')
#         for i in np.arange(start_year,end_year+1,1):
            
#             file = filename_start+variable+'_'+str(i)+'.nc'
#             files.append(file)
            
#         cubelist = iris.load(files)
#         cube = cubelist.concatenate()[0]
#         apply_common_time_coordinates(cube)
#         add_doy_month_year(cube)
#         add_2d_latlon_aux_coords(cube)
#         add_shifted_2d_latlon_aux_coords(cube)
#         cube.coord('grid_longitude').guess_bounds()
#         cube.coord('grid_latitude').guess_bounds()
#         cube.remove_coord('forecast_period')
#         if variable == '2mtemp':
#             cube.remove_coord('height')
#         cube.remove_coord('originating_centre')
#         cube.attributes = {}
        
#         cube.data = np.ma.asarray(cube.data, np.dtype('float32'))
        
#         if variable in ('total_precip', 'snowfall','snowmelt'):
#             cube.data = cube.data*10**3
#             cube = cube[3::4,:,:]
        
#     elif RCM =='ERA5':
#         os.chdir(input_path + 'Reanalysis_Data/ERA5/')
        
#         if variable == '2mtemp':
#             filename_start = 'era5_hourly_'
#         else:
#             filename_start = 'era5_'
            
#         for i in np.arange(start_year,end_year+1,1):
#             file = filename_start + variable + '_'+str(i)+'.grib'
#             files.append(file)
            
#         cubelist = iris.load(files)
#         cube = cubelist.concatenate()[0]
#         apply_common_time_coordinates(cube)
#         add_doy_month_year(cube)
#         add_2d_latlon_aux_coords(cube)
#         add_shifted_2d_latlon_aux_coords(cube)
#         cube.coord('grid_longitude').guess_bounds()
#         cube.coord('grid_latitude').guess_bounds()
#         cube.attributes = {}
#         cube.remove_coord('forecast_period')
#         if variable == '2mtemp':
#             cube.remove_coord('height')
#         cube.remove_coord('originating_centre')
#         cube.data = np.ma.asarray(cube.data, np.dtype('float32'))
        
#         #converting to 3hourly accumulation units and kg rather than mm. 
#         if variable in ('total_precip', 'snowfall','snowmelt'):
#             cube = convert_hourly_to_3hourly_accumulation(cube)
#             cube.data = cube.data*10**3
#     return(cube)

# def memory_efficient_loading(input_path,RCM,resolution,driving_data,start_year,end_year,variable,destination_path):
#     for i in tqdm(np.arange(start_year,end_year+1,1)):
#         cube = load_raw_data(input_path,RCM,resolution,driving_data,i,i,variable)
#         os.chdir(destination_path)
#         iris.save(cube, RCM+'_'+resolution+'_'+driving_data+'_'+variable+'_'+str(i)+'.nc') 

