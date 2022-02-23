# General Helper Functions

import cf_units as unit
import cftime
import iris
import iris.coord_categorisation
from iris.analysis.cartography import unrotate_pole
from iris.coords import AuxCoord
from iris.experimental.equalise_cubes import equalise_attributes
from iris.util import unify_time_units
import numpy as np
from numpy import meshgrid
from scipy.interpolate import griddata
from statsmodels.tsa.seasonal import STL
import pandas as pd
import dask
import dask.array as da
import dask.delayed as delayed

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def apply_common_time_coordinates(cube):
    u = unit.Unit('days since 1979-01-01 00:00:00', calendar='gregorian')
    
    if cube.coord('time').units.calendar == '365_day': # This is only needed for RACMO(ERA5)
        
        dates_365 = cube.coord('time').units.num2date(cube.coord('time').points)
        #iris.unit.num2date(time_value, unit, calendar) [For the given units and the given values, convert to datetimes]
        dates_gregorian = []

        for date in dates_365:
            year = date.year
            month = date.month
            day = date.day
            hour = date.hour
            dates_gregorian.append(cftime.DatetimeGregorian(year, month, day, hour, 0, 0, 0))
            
        # converts each 365 datetime object to a gregorian datetime object
        
        num_gregorian = unit.date2num(dates_gregorian, 'days since 1979-01-01 00:00:00', calendar='gregorian')
        
        # returns values for each datetime given units of gregorian and days since 1979
        
        cube.coord('time').points = num_gregorian # updates cubes time point values
        cube.coord('time').units = unit.Unit(u,calendar = 'gregorian') # updates cubes time units

    converted_points = cube.coord('time').units.convert(cube.coord('time').points,u,unit.FLOAT32)
    cube.coord('time').points = converted_points
    
    cube.coord('time').units = u
    cube.coord('time').var_name = 'time'
    cube.coord('time').long_name = 'time'
    cube.coord('time').standard_name = 'time'

    cube.coord('time').bounds = None
    cube.coord('time').guess_bounds()
    converted_bounds = cube.coord('time').units.convert(cube.coord('time').bounds,u,unit.FLOAT32)
    cube.coord('time').bounds = converted_bounds
            
def add_doy_month_year(cube):
    iris.coord_categorisation.add_day_of_year(cube, 'time', name='day_of_year')  
    iris.coord_categorisation.add_month(cube, 'time', name='month') 
    iris.coord_categorisation.add_year(cube, 'time', name='year')  

def add_2d_latlon_aux_coords(cube):
    rotated_grid_latitude = cube.coord('grid_latitude').points
    rotated_grid_longitude = cube.coord('grid_longitude').points
    lons,lats = meshgrid(rotated_grid_longitude, rotated_grid_latitude)
    cs = cube.coord_system()
    lons,lats = unrotate_pole(lons,lats, cs.grid_north_pole_longitude, cs.grid_north_pole_latitude)
    #lons,lats = rotate_pole(lons,lats, cs.grid_north_pole_longitude, cs.grid_north_pole_latitude)
    
    grid_lat_dim = cube.coord_dims('grid_latitude')[0]
    grid_lon_dim = cube.coord_dims('grid_longitude')[0]
    
    cube.add_aux_coord(AuxCoord(points=lats, standard_name='latitude', units='degrees'),(grid_lat_dim,grid_lon_dim))
    cube.add_aux_coord(AuxCoord(points=lons, standard_name='longitude', units='degrees'),(grid_lat_dim,grid_lon_dim))

def examine_postprocessed_data(filename,path):

    cube = iris.load(path+filename)[0]
    time = cube.coord('time')
    
    print(color.BOLD +color.PURPLE + filename + color.END)
    
    print(cube)
    
    print(color.BOLD + 'Data:' + color.END)
    print('Units =',cube.units)
    print('Mean =',cube.data.mean())
    print('Sum =',cube.data.sum())
    
    print(color.BOLD + 'Time:' + color.END)
    print('Start Date =',time.units.num2date(time.points[:1]))
    print('End Date =',time.units.num2date(time.points[-1:]))
    print('Units =',time.units)
    print('Frequency =',time.points[1]-time.points[0])

def regrid(cube,grid_cube,method):
    
    #Replacing masked values with zero value
    cube.data = cube.data.filled(0)
    
    #rotating coordinates to equator so distances are approximately euclidean 
    lons,lats = cube.coord('longitude').points,cube.coord('latitude').points
    rot_lons,rot_lats = unrotate_pole(lons,lats, 180, 0)    
    
    #sample points:
    points = np.array(list(zip(rot_lats.ravel(),rot_lons.ravel())))
    values = cube.data.ravel()
    
    #new grid points:
    grid_lons, grid_lats = grid_cube.coord('longitude').points,grid_cube.coord('latitude').points
    rot_grid_lons,rot_grid_lats = unrotate_pole(grid_lons,grid_lats, 180, 0)  
    
    #interpolating:
    regridded_data = griddata(points, values, (rot_grid_lats, rot_grid_lons), method=method, fill_value = 0)
       
    cube_regridded = iris.cube.Cube(
        regridded_data,
        long_name='cube_regridded',
        aux_coords_and_dims=[(grid_cube.coord('latitude'),grid_cube.coord_dims('latitude')),(grid_cube.coord('longitude'),grid_cube.coord_dims('longitude'))]
        )
    
    return(cube_regridded[:])

def remove_auxcoords(cube):
    for i in cube.aux_coords:
        cube.remove_coord(i)

def stl_decomposition(timeseries,start_date,frequency):
    data = timeseries[:]
    ds = pd.Series(data, index=pd.date_range(start_date, periods=len(data), freq=frequency), name = 'Decomposition')
    stl = STL(ds, seasonal=13)
    #stl = STL(ds, seasonal=13,robust=True)
    res = stl.fit()
    
    return ([res.trend,res.seasonal,res.resid]) 

def concatenate_cubes(cubelist):
    equalise_attributes(cubelist)
    unify_time_units(cubelist)
    return cubelist.concatenate_cube()

def multi_apply_along_axis(func1d, axis, arrs, *args, **kwargs):
    
    #arrs = np.copy(arrs)
    """
    Given a function `func1d(A, B, C, ..., *args, **kwargs)`  that acts on 
    multiple one dimensional arrays, apply that function to the N-dimensional
    arrays listed by `arrs` along axis `axis`
    
    If `arrs` are one dimensional this is equivalent to::
    
        func1d(*arrs, *args, **kwargs)
    
    If there is only one array in `arrs` this is equivalent to::
    
        numpy.apply_along_axis(func1d, axis, arrs[0], *args, **kwargs)
        
    All arrays in `arrs` must have compatible dimensions to be able to run
    `numpy.concatenate(arrs, axis)`
    
    Arguments:
        func1d:   Function that operates on `len(arrs)` 1 dimensional arrays,
                  with signature `f(*arrs, *args, **kwargs)`
        axis:     Axis of all `arrs` to apply the function along
        arrs:     Iterable of numpy arrays
        *args:    Passed to func1d after array arguments
        **kwargs: Passed to func1d as keyword arguments
    """
    # Concatenate the input arrays along the calculation axis to make one big
    # array that can be passed in to `apply_along_axis`
    carrs = np.concatenate(arrs, axis)
    
    # We'll need to split the concatenated arrays up before we apply `func1d`,
    # here's the offsets to split them back into the originals
    offsets=[]
    start=0
    for i in range(len(arrs)-1):
        start += arrs[i].shape[axis]
        offsets.append(start)
            
    # The helper closure splits up the concatenated array back into the components of `arrs`
    # and then runs `func1d` on them
    def helperfunc(a, *args, **kwargs):
        arrs = np.split(a, offsets)
        return func1d(*[*arrs, *args], **kwargs)
    
    # Run `apply_along_axis` along the concatenated array
    return np.apply_along_axis(helperfunc, axis, carrs, *args, **kwargs)

def RMSE(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

# def regrid_onto_regriddingcube(cube,regridding_cube):
    
#     if cube.coord('grid_latitude').bounds is None:
#         cube.coord('grid_latitude').guess_bounds()
        
#     if cube.coord('grid_longitude').bounds is None:
#         cube.coord('grid_longitude').guess_bounds()
        
#     if regridding_cube.coord('grid_latitude').bounds is None:
#         regridding_cube.coord('grid_latitude').guess_bounds()
        
#     if regridding_cube.coord('grid_longitude').bounds is None:
#         regridding_cube.coord('grid_longitude').guess_bounds()    
        
#     step1_scheme = iris.analysis.AreaWeighted(mdtol=1.0)
#     step2_scheme = iris.analysis.Linear(extrapolation_mode='mask')
    
#     #Step 1:
#     cube_adjusted_resolution = reducing_to_same_res(step1_scheme,cube,regridding_cube)
    
#     #Step 2:
#     cube_regridded = cube_adjusted_resolution.regrid(regridding_cube, step2_scheme)

    
#     add_2d_latlon_aux_coords(cube_regridded)
#     add_shifted_2d_latlon_aux_coords(cube_regridded)
#     cube_regridded.coord('latitude').var_name = 'latitude'
#     cube_regridded.coord('longitude').var_name = 'longitude'
#     cube_regridded.coord('shifted_latitude').var_name = 'shifted_latitude'
#     cube_regridded.coord('shifted_longitude').var_name = 'shifted_longitude'
    
#     return(cube_regridded)     
    
# def aggregate_to_daily(cube):
#     iris.coord_categorisation.add_year(cube, 'time', name='year')
#     iris.coord_categorisation.add_day_of_year(cube, 'time', name='day_of_year')
#     return cube.aggregated_by(['year','day_of_year'],iris.analysis.MEAN)

# def concatenate_cubes(cubelist):
#     equalise_attributes(cubelist)
#     unify_time_units(cubelist)
#     return cubelist.concatenate_cube()

# def mar_add_2d_latlon_coordinates(cube):
#     y_values = cube.coord('y').points*1000
#     x_values = cube.coord('x').points*1000
#     xs, ys = meshgrid(x_values,y_values)
    
#     inProj = Proj(init='epsg:3031')
#     outProj = Proj(init='epsg:4326')
#     lons, lats = transform(inProj,outProj,xs,ys)

#     y_dim = cube.coord_dims('y')[0]
#     x_dim = cube.coord_dims('x')[0]
    
#     cube.add_aux_coord(AuxCoord(points=lats, standard_name='latitude', units='degrees'),(y_dim,x_dim))
#     cube.add_aux_coord(AuxCoord(points=lons, standard_name='longitude', units='degrees'),(y_dim,x_dim))
    
#     return cube

# def add_shifted_2d_latlon_aux_coords(cube):
#     rotated_grid_latitude = cube.coord('grid_latitude').points
#     rotated_grid_longitude = cube.coord('grid_longitude').points
#     half_latitude_diff = (rotated_grid_latitude[1]-rotated_grid_latitude[0])/2
#     half_longitude_diff = (rotated_grid_longitude[1]-rotated_grid_longitude[0])/2
#     shifted_rotated_grid_latitude = rotated_grid_latitude - half_latitude_diff
#     shifted_rotated_grid_longitude = rotated_grid_longitude - half_longitude_diff
    
#     cs = cube.coord_system()
#     lons,lats = meshgrid(shifted_rotated_grid_longitude, shifted_rotated_grid_latitude)
#     lons,lats = unrotate_pole(lons,lats, cs.grid_north_pole_longitude, cs.grid_north_pole_latitude)
    
#     grid_lat_dim = cube.coord_dims('grid_latitude')[0]
#     grid_lon_dim = cube.coord_dims('grid_longitude')[0]

#     cube.add_aux_coord(AuxCoord(points=lats, long_name='shifted_latitude', units='degrees'),(grid_lat_dim,grid_lon_dim))
#     cube.add_aux_coord(AuxCoord(points=lons, long_name='shifted_longitude', units='degrees'),(grid_lat_dim,grid_lon_dim))        
    
# def ice_shelf_filter(cube,ice_shelf):
#     #ice_shelf needs to be a shapely geometry e.g. gdf.loc[44].geometry
#     longitude_points = cube.coord('longitude').points
#     latitude_points = cube.coord('latitude').points
#     ice_shelf_filter = np.empty(cube.shape,dtype=bool)
    
#     for i in tqdm(np.arange(0,cube.shape[0])):
#         for j in np.arange(0,cube.shape[1]):
#             if ice_shelf.contains(Point(longitude_points[i,j],latitude_points[i,j])):
#                 ice_shelf_filter[i,j]=True
#             else:
#                 ice_shelf_filter[i,j]=False
                
#     filtered_cube = cube[:]
#     mask_data = np.broadcast_to(ice_shelf_filter,cube.shape)
#     #filtered_cube.data[ice_shelf_filter==False]=np.nan
#     filtered_cube.data = np.ma.masked_where(mask_data==False,filtered_cube.data)
#     return filtered_cube

# def cubelist_of_filtered_geometries(cube,geodf):
#     cubelist = []
#     for i in np.arange(0,geodf.shape[0]):
#         iceshelf = geodf.iloc[i].geometry
#         filtered_cube = ice_shelf_filter(cube,iceshelf)
#         cubelist.append(filtered_cube)
#     return iris.cube.CubeList(cubelist)

# def filtered_box(cube):

#     condition = np.isnan(cube.data)

#     grid_lat=cube.coord('grid_latitude').points
#     grid_lon=cube.coord('grid_longitude').points

#     cube_x_axis,cube_y_axis = np.meshgrid(grid_lat,grid_lon,indexing='ij')

#     reduced_grid_lats = cube_x_axis[condition==False]
#     reduced_grid_lons = cube_y_axis[condition==False]

#     min_grid_lat = reduced_grid_lats.min()
#     min_grid_lat_index = np.argwhere(grid_lat==min_grid_lat)[0][0]
#     max_grid_lat = reduced_grid_lats.max()
#     max_grid_lat_index = np.argwhere(grid_lat==max_grid_lat)[0][0]

#     min_grid_lon = reduced_grid_lons.min()
#     min_grid_lon_index = np.argwhere(grid_lon==min_grid_lon)[0][0]
#     max_grid_lon = reduced_grid_lons.max()
#     max_grid_lon_index = np.argwhere(grid_lon==max_grid_lon)[0][0]
    
#     grid_lat_indexes = np.arange(min_grid_lat_index,max_grid_lat_index,1)
#     grid_lon_indexes = np.arange(min_grid_lon_index,max_grid_lon_index,1)
    
#     line_1_lons = cube[grid_lat_indexes[0],grid_lon_indexes].coord('longitude').points
#     line_1_lats = cube[grid_lat_indexes[0],grid_lon_indexes].coord('latitude').points
#     line_2_lons = cube[grid_lat_indexes,grid_lon_indexes[-1]].coord('longitude').points
#     line_2_lats = cube[grid_lat_indexes,grid_lon_indexes[-1]].coord('latitude').points
#     line_3_lons = cube[grid_lat_indexes[-1],grid_lon_indexes[::-1]].coord('longitude').points
#     line_3_lats = cube[grid_lat_indexes[-1],grid_lon_indexes[::-1]].coord('latitude').points
#     line_4_lons = cube[grid_lat_indexes[::-1],grid_lon_indexes[0]].coord('longitude').points
#     line_4_lats = cube[grid_lat_indexes[::-1],grid_lon_indexes[0]].coord('latitude').points

#     box_lons = list(line_1_lons) + list(line_2_lons) + list(line_3_lons) + list(line_4_lons)
#     box_lats = list(line_1_lats) + list(line_2_lats) + list(line_3_lats) + list(line_4_lats)

#     return([box_lons,box_lats])
    
# def boundary_box(cube):
#     #Requires 2D cube with dimensions grid latitude and grid longitude as well as 2D lat&lon auxiliary coordinates
#     line_1_lons = cube[0,:].coord('longitude').points
#     line_1_lats = cube[0,:].coord('latitude').points
#     line_2_lons = cube[:,-1].coord('longitude').points
#     line_2_lats = cube[:,-1].coord('latitude').points
#     line_3_lons = cube[-1,::-1].coord('longitude').points
#     line_3_lats = cube[-1,::-1].coord('latitude').points
#     line_4_lons = cube[::-1,0].coord('longitude').points
#     line_4_lats = cube[::-1,0].coord('latitude').points

#     box_lons = list(line_1_lons) + list(line_2_lons) + list(line_3_lons) + list(line_4_lons)
#     box_lats = list(line_1_lats) + list(line_2_lats) + list(line_3_lats) + list(line_4_lats)
    
#     box_lons_lats = [box_lons,box_lats]
    
#     return box_lons_lats
    
# def reducing_to_075_res(cube,scheme):
#     grid_lat = cube.coord('grid_latitude')[:]
#     grid_lon = cube.coord('grid_longitude')[:]
    
#     new_grid_lat = iris.coords.DimCoord(np.arange(grid_lat.points.min(),grid_lat.points.max(),0.75), standard_name='grid_latitude',units='degrees')
#     new_grid_lon = iris.coords.DimCoord(np.arange(grid_lon.points.min(),grid_lon.points.max(),0.75), standard_name='grid_longitude',units='degrees')
    
#     new_grid_lat.coord_system = grid_lat.coord_system
#     new_grid_lon.coord_system = grid_lon.coord_system
    
#     grid_latitude = new_grid_lat
#     grid_longitude = new_grid_lon
    
#     new_grid_cube = iris.cube.Cube(
#      np.zeros((len(new_grid_lat.points),len(new_grid_lon.points)),np.float32),
#      long_name='Common_Grid',
#      dim_coords_and_dims=[(grid_latitude,0),(grid_longitude,1)])
        
#     new_grid_cube.coord('grid_latitude').guess_bounds()
#     new_grid_cube.coord('grid_longitude').guess_bounds()

#     time_coord_len = len(cube.coord('time').points)
#     regridder = scheme.regridder(cube[0], new_grid_cube)
#     regridder_cubelist = []

#     for i in tqdm(np.arange(0,time_coord_len,1000)):
#         if i+1000>time_coord_len:
#             regridded_cube = regridder(cube[i:time_coord_len])
#             regridder_cubelist.append(regridded_cube)
#         else:
#             regridded_cube = regridder(cube[i:i+1000])
#             regridder_cubelist.append(regridded_cube)

#     cubelist = iris.cube.CubeList(regridder_cubelist)
#     return(concatenate_cubes(cubelist)[0])

# def reducing_to_same_res(scheme,cube,regridding_cube):
    
#     lat_resolution = np.abs(regridding_cube.coord('grid_latitude').points[0]-regridding_cube.coord('grid_latitude').points[1])
#     lon_resolution = np.abs(regridding_cube.coord('grid_longitude').points[0]-regridding_cube.coord('grid_longitude').points[1])
    
#     grid_lat = cube.coord('grid_latitude')[:]
#     grid_lon = cube.coord('grid_longitude')[:]
    
#     new_grid_lat = iris.coords.DimCoord(np.arange(grid_lat.points.min(),grid_lat.points.max(),lat_resolution), standard_name='grid_latitude',units='degrees')
#     new_grid_lon = iris.coords.DimCoord(np.arange(grid_lon.points.min(),grid_lon.points.max(),lon_resolution), standard_name='grid_longitude',units='degrees')
    
#     #new_grid_lat = iris.coords.DimCoord(np.arange(grid_lat.points.min()-lat_resolution,grid_lat.points.max()+lat_resolution,lat_resolution), standard_name='grid_latitude',units='degrees')
#     #new_grid_lon = iris.coords.DimCoord(np.arange(grid_lon.points.min()-lon_resolution,grid_lon.points.max()+lon_resolution,lon_resolution), standard_name='grid_longitude',units='degrees')
    
#     new_grid_lat.coord_system = grid_lat.coord_system
#     new_grid_lon.coord_system = grid_lon.coord_system
    
#     grid_latitude = new_grid_lat
#     grid_longitude = new_grid_lon
    
#     new_grid_cube = iris.cube.Cube(
#      np.zeros((len(new_grid_lat.points),len(new_grid_lon.points)),np.float32),
#      long_name='Common_Grid',
#      dim_coords_and_dims=[(grid_latitude,0),(grid_longitude,1)])
        
#     new_grid_cube.coord('grid_latitude').guess_bounds()
#     new_grid_cube.coord('grid_longitude').guess_bounds()
    
#     regridder_cubelist = []
    

#     if cube.coord('time').shape == (1,):
#         regridded_cube = cube.regrid(new_grid_cube, scheme)
#         regridder_cubelist.append(regridded_cube)

#     else:
#         time_coord_len = len(cube.coord('time').points)
#         regridder = scheme.regridder(cube[0], new_grid_cube)


#         for i in tqdm(np.arange(0,time_coord_len,1000)):
#             if i+1000>time_coord_len:
#                 regridded_cube = regridder(cube[i:time_coord_len])
#                 regridder_cubelist.append(regridded_cube)
#             else:
#                 regridded_cube = regridder(cube[i:i+1000])
#                 regridder_cubelist.append(regridded_cube)

#     cubelist = iris.cube.CubeList(regridder_cubelist)

#     return(concatenate_cubes(cubelist))
        
# def round_time_coord_nearest_0_5(cube):
#     time_coord = cube.coord('time')
#     time_coord.points = np.around(time_coord.points * 2) / 2.0
#     time_coord.bounds=None
#     time_coord.guess_bounds()
    
# def apply_common_mask(list_of_cubes):
    
#     list_of_cubes=list_of_cubes[:]
    
#     masked_list = []
#     mask_array = list_of_cubes[0][0].data
    
#     for i in np.arange(1,len(list_of_cubes),1):
#         mask_array = mask_array - list_of_cubes[i][0].data
        
#     for i in np.arange(0,len(list_of_cubes),1):
#         cube = list_of_cubes[i][:]
#         mask_array_broadcasted = np.broadcast_to(mask_array, cube.shape,subok=True)
#         mask_array_broadcasted.mask = np.broadcast_to(mask_array.mask, cube.shape,subok=True)
#         cube.data = np.ma.array(cube[:].data,mask = mask_array_broadcasted.mask)
        
#         masked_list.append(cube)
    
#     return(masked_list)

# def apply_common_time_coord(list_of_cubes):
    
#     list_of_cubes=list_of_cubes[:]
    
#     list_of_common_time_coord_cubes = []

#     common_time_coord_points = None

#     counter =0
#     for cube in list_of_cubes:
#         if counter == 0:
#             common_time_coord_points = set(cube.coord('time').points)
#             counter+=1
#         else:
#             common_time_coord_points &= set(cube.coord('time').points)
        
#     for cube in list_of_cubes:
        
#         boolean_array = np.array([x in common_time_coord_points for x in cube.coord('time').points])
        
#         cube = cube[boolean_array]
        
#         list_of_common_time_coord_cubes.append(cube)
    
#     return(list_of_common_time_coord_cubes)

# def aggregate_to_daily(cube):
#     cube=cube[:]
    
#     daily_cube = cube.aggregated_by(['year','day_of_year'], iris.analysis.MEAN)
#     round_time_coord_nearest_0_5(daily_cube)
    
#     return(daily_cube)

# def calculate_ensemble_mean(list_of_cubes):

#     list_of_cubes=list_of_cubes[:]

#     ensemble_list = []
    
#     for i in np.arange(0,len(list_of_cubes),1):
#         cube = list_of_cubes[i][:]
#         cube.remove_coord('month')
#         cube.remove_coord('year')
#         cube.remove_coord('day_of_year')
        
#         #cube.remove_coord('grid_latitude')
#         #cube.remove_coord('grid_longitude')
        
#         cube.cell_methods = None
#         cube.long_name = '2mtemp'
#         cube.var_name = '2mtemp'
#         cube.standard_name = 'air_temperature'
#         model = iris.coords.AuxCoord(i, long_name='model', units='no_unit')
#         cube.add_aux_coord(model)
#         ensemble_list.append(cube)
    
#     iriscubelist = iris.cube.CubeList(ensemble_list)

#     ensemble = iriscubelist.merge_cube()
#     ensemble_mean = ensemble.collapsed('model',iris.analysis.MEAN)
    
#     return(ensemble_mean)

# def calculate_ensemble_stddev(list_of_cubes):

#     list_of_cubes=list_of_cubes[:]

#     ensemble_list = []
    
#     for i in np.arange(0,len(list_of_cubes),1):
#         cube = list_of_cubes[i][:]
#         cube.remove_coord('month')
#         cube.remove_coord('year')
#         cube.remove_coord('day_of_year')
        
#         #cube.remove_coord('grid_latitude')
#         #cube.remove_coord('grid_longitude')
        
#         cube.cell_methods = None
#         cube.long_name = '2mtemp'
#         cube.var_name = '2mtemp'
#         cube.standard_name = 'air_temperature'
#         model = iris.coords.AuxCoord(i, long_name='model', units='no_unit')
#         cube.add_aux_coord(model)
#         ensemble_list.append(cube)
    
#     iriscubelist = iris.cube.CubeList(ensemble_list)

#     ensemble = iriscubelist.merge_cube()
#     ensemble_stddev = ensemble.collapsed('model',iris.analysis.STD_DEV)

#     #return(ensemble)
#     return(ensemble_stddev)

# def calculate_relative_cubes(list_of_cubes):
    
#     list_of_cubes=list_of_cubes[:]

#     relative_cubes = []
      
#     ensemble_mean = calculate_ensemble_mean(list_of_cubes)
    
#     for i in np.arange(0,len(list_of_cubes),1):
#         cube = list_of_cubes[i]
#         cube.remove_coord('day_of_year')
#         cube.remove_coord('year')
#         cube.remove_coord('month')
#         relative_cube = cube-ensemble_mean
#         relative_cubes.append(relative_cube)
    
#     return(relative_cubes) 

# def calculate_stddev_cube(list_of_cubes):
    
#     list_of_cubes=list_of_cubes[:]

#     relative_cubes = []
      
#     ensemble_mean = calculate_ensemble_mean(list_of_cubes)
    
#     for i in np.arange(0,len(list_of_cubes),1):
#         cube = list_of_cubes[i]
#         cube.remove_coord('day_of_year')
#         cube.remove_coord('year')
#         cube.remove_coord('month')
#         relative_cube = cube-ensemble_mean
#         relative_cubes.append(relative_cube)
    
#     return(relative_cubes) 

# def mask_to_land_only(cube, mask_cube):
    
#     cube = cube[:]
#     mask_cube = mask_cube[:]

#     print(f'Time: {time.time() - start}')
    
    
#     mask_cube = era5_land_sea_mask[:]
#     print(f'Time: {time.time() - start}')

#     #mask_cube.data = np.ma.masked_where(mask_cube.data < 0.5,mask_cube.data)
#     mask_cube_cg = regrid_onto_regriddingcube(mask_cube,regridding_cube)
#     print(f'Time: {time.time() - start}')

#     mask_cube_cg.data = np.ma.masked_where(mask_cube_cg.data < 0.5,mask_cube_cg.data)
#     print(f'Time: {time.time() - start}')

#     mask_array = mask_cube_cg.data
    
    
#     mask_array_broadcasted = np.broadcast_to(mask_array, cube.shape,subok=True)
#     mask_array_broadcasted.mask = np.broadcast_to(mask_array.mask, cube.shape,subok=True)
    
#     cube.data = np.ma.array(cube.data[:],mask = mask_array_broadcasted.mask)
#     #cube.data = np.ma.array(cube.data[:],mask = mask_array_broadcasted.mask)

    
#     #for i in np.arange(0,cube.data.shape[0],1):
#      #   cube[i].data = np.ma.array(cube[i][:].data,mask = mask_array_broadcasted.mask[i])
        
#     return(cube)

# def mask_to_land_only(cube, regridding_cube):
    
#     cube = cube[:]
    
#     os.chdir(os.path.expanduser("~"))
#     era5_land_sea_mask = iris.load('/home/carter10/data/era5_land_sea_mask.grib')[0]

#     start = time.time()

#     print(f'Time: {time.time() - start}')
    
    
#     mask_cube = era5_land_sea_mask[:]
#     print(f'Time: {time.time() - start}')

#     #mask_cube.data = np.ma.masked_where(mask_cube.data < 0.5,mask_cube.data)
#     mask_cube_cg = regrid_onto_regriddingcube(mask_cube,regridding_cube)
#     print(f'Time: {time.time() - start}')

#     mask_cube_cg.data = np.ma.masked_where(mask_cube_cg.data < 0.5,mask_cube_cg.data)
#     print(f'Time: {time.time() - start}')

#     mask_array = mask_cube_cg.data
    
    
#     mask_array_broadcasted = np.broadcast_to(mask_array, cube.shape,subok=True)
#     mask_array_broadcasted.mask = np.broadcast_to(mask_array.mask, cube.shape,subok=True)
    
#     cube.data = np.ma.array(cube.data[:],mask = mask_array_broadcasted.mask)
#     #cube.data = np.ma.array(cube.data[:],mask = mask_array_broadcasted.mask)

    
#     #for i in np.arange(0,cube.data.shape[0],1):
#      #   cube[i].data = np.ma.array(cube[i][:].data,mask = mask_array_broadcasted.mask[i])
        
#     return(cube)

# def convert_hourly_to_3hourly_accumulation(cube):
    
#     cube = cube[:]
    
#     def category_function(coord,point):
#         return (np.int(point*24/3)+1)*3/24
    
#     units = cube.coord('time').units
    
#     iris.coord_categorisation.add_categorised_coord(cube, '3htime', 'time', category_function, units=cube.coord('time').units)
    
#     cube = cube.aggregated_by('3htime',iris.analysis.SUM)
    
#     cube.coord('time').points = cube.coord('3htime').points
    
#     cube.coord('time').bounds = None
#     cube.coord('time').guess_bounds()
    
#     cube.remove_coord('3htime')
    
#     return cube  
    
# def retrieve_resolution_in_km(cube):
#     deg_res = cube.coord('grid_latitude').points[1]-cube.coord('grid_latitude').points[0]
#     km_res = (deg_res/360)*40075
#     return(km_res)

# def remove_auxcoords(cube):
#     for i in cube.aux_coords:
#         cube.remove_coord(i)


# def mergeDict(dict1, dict2):
#     ''' Merge dictionaries and keep values of common keys in list'''
#     dict3 = {**dict1, **dict2}
    
#     for key, value in dict3.items():
#         if key in dict1 and key in dict2:
#             if isinstance(dict1[key], list):
#                 alist = dict1[key][:]
#                 alist.append(dict2[key])
#                 dict3[key] = alist

#             else:
#                 dict3[key] = [dict1[key] , dict2[key]]
    
#     return dict3

# def fix_polygon_discontinuity_over_pole(polygon):
#     boundary = np.array(polygon.boundary)
#     i = 0
#     while i < boundary.shape[0] - 1:
#         if abs(boundary[i+1,0] - boundary[i,0]) > 180:
#             assert (boundary[i,1] > 0) == (boundary[i,1] > 0)
#             vsign = -1 if boundary[i,1] < 0 else 1
#             hsign = -1 if boundary[i,0] < 0 else 1
#             boundary = np.insert(boundary, i+1, [
#                 [hsign*179.9999, boundary[i,1]],
#                 [hsign*179.9999, vsign*89.9999],
#                 [-hsign*179.999, vsign*89.9999],
#                 [-hsign*179.999, boundary[i+1,1]]
#             ], axis=0)
#             i += 5
#         else:
#             i += 1
#     return(Polygon(boundary))

# def filter_cube_by_domain(cube,domain_cube):
#     filtered_cube = cube[:]
#     rotated_domain_cube = domain_cube[:]
    
#     #work with rotated lat,lon values to avoid pole problems
#     lons,lats = cube.coord('longitude').points,cube.coord('latitude').points
#     rot_lons,rot_lats = unrotate_pole(lons,lats, 180, 0)    
    
#     domain_lons,domain_lats = domain_cube.coord('longitude').points,domain_cube.coord('latitude').points
#     rot_domain_lons,rot_domain_lats = unrotate_pole(domain_lons,domain_lats, 180, 0)    
    
#     rotated_domain_cube.coord('longitude').points = rot_domain_lons
#     rotated_domain_cube.coord('latitude').points = rot_domain_lats

#     polygon = Polygon(np.column_stack((boundary_box(rotated_domain_cube)[0], boundary_box(rotated_domain_cube)[1])))
#     #fixed_polygon = fix_polygon_discontinuity_over_pole(polygon)
    
#     filter_array = np.zeros(rot_lons.shape)
    
#     for i in (np.arange(0,rot_lons.shape[0])):
#         for j in np.arange(0,rot_lons.shape[1]):
#             if polygon.contains(Point(rot_lons[i,j],rot_lats[i,j])):
#                 filter_array[i,j]=True
    
#     mask_array = np.broadcast_to(filter_array,filtered_cube.shape)
#     filtered_cube.data = np.ma.masked_where(mask_array==False,filtered_cube.data)
#     return filtered_cube

# def joint_landsea_mask(mask1,mask2,diff_threshold,mask_threshold,method):
#     mask1_cp = mask1[:]
#     mask2_cp = mask2[:]
    
#     mask1_cp = regrid(mask1_cp,mask2_cp,method)
    
#     abs_difference = np.abs(mask2_cp.data-mask1_cp.data)
    
#     mask2_cp.data = np.ma.masked_where(abs_difference>diff_threshold,mask2_cp.data)
#     mask2_cp.data = np.ma.masked_where(mask2_cp.data<mask_threshold,mask2_cp.data)
    
#     return(mask2_cp)

# def load_all_years(filename_no_year,start_year,end_year,aggregation):
    
#     os.chdir(os.path.expanduser("~"))
#     os.chdir('Shared_Storage/Google_Bucket_Transfer/Postprocessed_Data')
    
#     all_years_list = []
    
#     for i in tqdm(np.arange(start_year,end_year+1,1)):
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

#         all_years_list.append(cube)
        
#     all_years_cubelist = iris.cube.CubeList(all_years_list)
    
#     for cube in all_years_cubelist:
#         cube.cell_methods = None

#     cube = all_years_cubelist.merge_cube()
    
#     return(cube)

