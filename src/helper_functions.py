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
import numpy.ma as ma
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
    
def regriding_impact(filename,paths,mask):
    broadcasted_mask = np.broadcast_to(mask[np.newaxis,:,:],[12,392,504])
    
    for path in paths:
        cube = iris.load(path+filename)[0]
        cube.data = cube.data*12250**2/10**12
        cube.data = ma.masked_where(broadcasted_mask==False, cube.data) # masking to land only
        print(color.BOLD +color.PURPLE + path+filename + color.END)  
        print('Mean =',cube.data.mean())
        print('Sum =',cube.data.sum())

def regrid(cube,grid_cube,method):
    
    if isinstance(cube.data, ma.MaskedArray):
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

