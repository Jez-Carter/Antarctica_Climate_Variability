# Plotting Functions:

import os
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

def pcolormesh_basemapplot(cube,basemap,vmin,vmax,cmap=None,alpha=1):

    longitudes = cube.coord('longitude').points
    latitudes = cube.coord('latitude').points

    current_dir = os.getcwd()
    os.chdir(os.path.expanduser("~"))
    os.chdir('Shared_Storage/Google_Bucket_Transfer/Ice_Shelf_Mask_Antarctica')
    basemap.readshapefile('antarctica_shapefile', 'antarctica_shapefile',linewidth=0.1,antialiased=False,color='k')
    os.chdir(current_dir)
    
    return(basemap.pcolormesh(longitudes,latitudes,cube.data,vmin=vmin,vmax=vmax, latlon=True, cmap=cmap, shading = 'nearest',alpha=alpha))

def median_correlation_plot(input_path,variables,component,mask,mean_melt,basemap,grid_cube):
    broadcasted_mean_melt = np.broadcast_to(mean_melt[np.newaxis,np.newaxis,:,:],[28,2,392,504])
    broadcasted_mask = np.broadcast_to(mask[np.newaxis,np.newaxis,:,:],[28,2,392,504])
    counter = 0
    for variable in variables:
        corr_data = np.load(f'{input_path}ensemble_stl_decomp_corr_{variable}.npy') # has shape [64,3,392,504]
        corr_data = corr_data[:,component,:,:] # filtering to component of time series
        corr_data = corr_data.reshape((8,8,2,392,504))
        corr_data = corr_data[np.triu_indices(8, k = 1)] #filtering to only unique correlation pairs, produces shape [28,2,392,504]
        if variable == 'melt':
            corr_data = ma.masked_where(broadcasted_mask==False, corr_data)
            corr_data = ma.masked_where(broadcasted_mean_melt <= 1, corr_data)
        median_corr = np.nanmedian(corr_data,axis=0) # returns the median value of the 28 unique correlation pairs.
        
        grid_cube.data = median_corr[0]
        plt.subplot(1, len(variables), counter+1)
        vmin=0
        vmax=1
        pcolormesh_basemapplot(grid_cube,basemap,vmin,vmax,cmap='Spectral')
        counter += 1
        
def difference_to_ensemble_plot(input_path,basemap,vmin,vmax,variable,models,grid_cube,mask,mean_melt):
    broadcasted_mean_melt = np.broadcast_to(mean_melt[np.newaxis,np.newaxis,:,:],[3,len(models),392,504])
    broadcasted_mask = np.broadcast_to(mask[np.newaxis,np.newaxis,:,:],[3,len(models),392,504])
    
    mean = np.load(f'{input_path}ensemble_stl_decomp_{variable}_mean.npy') #has shape [8,392,504]
    trend_stddev = np.load(f'{input_path}ensemble_stl_decomp_{variable}_trend_stddev.npy') #has shape [8,392,504]
    seasonal_stddev = np.load(f'{input_path}ensemble_stl_decomp_{variable}_seasonal_stddev.npy') #has shape [8,392,504]
    residual_stddev = np.load(f'{input_path}ensemble_stl_decomp_{variable}_residual_stddev.npy') #has shape [8,392,504]
    data = np.array((mean,seasonal_stddev,residual_stddev)) # data has shape [3,8,392,504]
    data = data[:,models]
    ensemble_mean = data.mean(axis=1)
    data = data - np.broadcast_to(ensemble_mean[:,np.newaxis,:,:],[3,len(models),392,504])
    # data has shape [3,len(models),392,504]
    
    if variable in ['snowfall','melt']:
        ensemble_mean_trend_stddev = trend_stddev.mean(axis=0) # shape [392,504]
        data = data/np.broadcast_to(ensemble_mean_trend_stddev[np.newaxis,np.newaxis,:,:],[3,len(models),392,504]) # turning into proportional difference
    
    if variable == 'melt':
        data = ma.masked_where(broadcasted_mask==False, data)
        data = ma.masked_where(broadcasted_mean_melt <= 1, data)
    
    for i in np.arange(0,len(models),1):
        for j in np.arange(0,3,1):
            plt_number = i*3+j+1
            plt.subplot(len(models), 3, plt_number)
            grid_cube.data = data[j,i]
            pcolormesh_basemapplot(grid_cube,basemap,vmin,vmax,cmap='Spectral_r')

# def contour_basemapplot(cube,basemap,levels,cmap=None,alpha=1):

#     longitudes = cube.coord('longitude').points
#     latitudes = cube.coord('latitude').points
    
#     return(basemap.contour(longitudes,latitudes,cube.data, latlon=True,levels=levels, cmap=cmap, shading = 'nearest',alpha=alpha))

# def pcolormesh_basemapplot_edge(cube,basemap,alpha,linewidth,cmap='BuPu',label=None):

#     longitudes = cube.coord('longitude').points
#     latitudes = cube.coord('latitude').points

#     current_dir = os.getcwd()
#     os.chdir(os.path.expanduser("~"))
#     os.chdir('Shared_Storage/Google_Bucket_Transfer/Ice_Shelf_Mask_Antarctica')
#     basemap.readshapefile('antarctica_shapefile', 'antarctica_shapefile',linewidth=0.1,antialiased=False,color='k')
#     os.chdir(current_dir)
    
#     return(basemap.pcolormesh(longitudes,latitudes,cube.data, latlon=True, cmap=cmap, shading = 'nearest', edgecolor='k',alpha=alpha,linewidth=linewidth,label=label))
 
# def pcolormesh_basemapplot_norm(cube,basemap,vmin,vmax,cmap=None):

#     longitudes = cube.coord('longitude').points
#     latitudes = cube.coord('latitude').points

#     current_dir = os.getcwd()
#     os.chdir(os.path.expanduser("~"))
#     os.chdir('Shared_Storage/Google_Bucket_Transfer/Ice_Shelf_Mask_Antarctica')
#     basemap.readshapefile('antarctica_shapefile', 'antarctica_shapefile')
#     os.chdir(current_dir)
    
#     #return(basemap.pcolormesh(longitudes,latitudes,cube.data,vmin=vmin,vmax=vmax, latlon=True, cmap=cmap, shading = 'nearest'))
#     if vmin>=0:
#         norm=colors.LogNorm(vmin=vmin, vmax=vmax)
#     elif vmin<0:
#         norm=colors.SymLogNorm(vmin=vmin, vmax=vmax, linthresh = 2, linscale=3)
#     return(basemap.pcolormesh(longitudes,latitudes,cube.data,norm=norm, latlon=True, cmap=cmap, shading = 'nearest'))
    
# def plot_boundary_box(box,basemap,color,label,linewidth,alpha):
#     lons = box[0]
#     lats = box[1]
#     x, y = antarctica_map(lons, lats)
#     basemap.plot(x, y, marker=None,color=color,label=label,linewidth=linewidth,alpha=alpha)
    
# def save_plot(plt_name,plt_resolution,plt_format):

#     #example: plt_name='test_plot', plt_resolution=300, plt_format='tight'
#     # plt_name could also contain a folder path

#     current_dir = os.getcwd()
#     os.chdir(os.path.expanduser("~"))
#     os.chdir('Shared_Storage/Google_Bucket_Transfer/Results')
#     plt.savefig(plt_name, dpi=plt_resolution, bbox_inches=plt_format)
#     os.chdir(current_dir)


# # In[8]:


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


# # In[9]:


# def load_all_years(filename_no_year,start_year,end_year):
    
#     #os.chdir(os.path.expanduser("~"))
#     #os.chdir('Shared_Storage/Google_Bucket_Transfer/Postprocessed_Data')
    
#     all_years_list = []
    
#     for i in tqdm(np.arange(start_year,end_year+1,1)):
#         cube = iris.load(filename_no_year+'_'+str(i)+'.nc')[0]
#         new_coord = iris.coords.AuxCoord(i, long_name='year', units='no_unit')
#         cube.add_aux_coord(new_coord)
#         all_years_list.append(cube)
        
#     all_years_cubelist = iris.cube.CubeList(all_years_list)
    
#     for cube in all_years_cubelist:
#         cube.cell_methods = None

#     cube = all_years_cubelist.merge_cube()
    
#     return(cube)