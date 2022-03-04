# Plotting Functions:

import os
import numpy as np
import numpy.ma as ma
import pandas as pd
pd.set_option('display.precision',2)
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr

def pcolormesh_basemapplot(cube,basemap,vmin,vmax,cmap=None,alpha=1):

    longitudes = cube.coord('longitude').points
    latitudes = cube.coord('latitude').points

    current_dir = os.getcwd()
    os.chdir(os.path.expanduser("~"))
    os.chdir('Shared_Storage/Google_Bucket_Transfer/Ice_Shelf_Mask_Antarctica')
    basemap.readshapefile('antarctica_shapefile', 'antarctica_shapefile',linewidth=0.1,antialiased=False,color='k')
    os.chdir(current_dir)
    
    return(basemap.pcolormesh(longitudes,latitudes,cube.data,vmin=vmin,vmax=vmax, latlon=True, cmap=cmap, shading = 'nearest',alpha=alpha))

def median_correlation_plot(input_path,variables,component,mask,mean_melt,basemap,grid_cube,subplot_dimensions,plt_numbers,vmins,vmaxs):
    plots= []
    broadcasted_mean_melt = np.broadcast_to(mean_melt[np.newaxis,np.newaxis,:,:],[28,2,392,504])
    broadcasted_mask = np.broadcast_to(mask[np.newaxis,np.newaxis,:,:],[28,2,392,504])
    for variable,plt_number,vmin,vmax in zip(variables,plt_numbers,vmins,vmaxs):
        corr_data = np.load(f'{input_path}ensemble_stl_decomp_corr_{variable}.npy') # has shape [64,3,2,392,504]
        corr_data = corr_data[:,component,:,:] # filtering to component of time series
        corr_data = corr_data.reshape((8,8,2,392,504))
        corr_data = corr_data[np.triu_indices(8, k = 1)] #filtering to only unique correlation pairs, produces shape [28,2,392,504]
        if variable == 'melt':
            corr_data = ma.masked_where(broadcasted_mask==False, corr_data)
            corr_data = ma.masked_where(broadcasted_mean_melt <= 1, corr_data)
        median_corr = np.nanmedian(corr_data,axis=0) # returns the median value of the 28 unique correlation pairs.
        
        grid_cube.data = median_corr[0]
        plt.subplot(subplot_dimensions[0], subplot_dimensions[1], plt_number)
        plot = pcolormesh_basemapplot(grid_cube,basemap,vmin,vmax,cmap='viridis')
        plots.append(plot)
    return(plots)

def correlation_scatter_plot(input_path,variables,component,mask,mean_melt,subplot_dimensions,plt_numbers,scale,vmins):
    plots = []
    broadcasted_mean_melt = np.broadcast_to(mean_melt[np.newaxis,np.newaxis,:,:],[8,8,392,504])
    broadcasted_mask = np.broadcast_to(mask[np.newaxis,np.newaxis,:,:],[8,8,392,504])
    
    for variable,plt_number,vmin in zip(variables,plt_numbers,vmins):
        corr_data = np.load(f'{input_path}ensemble_stl_decomp_corr_{variable}.npy') # has shape [64,3,2,392,504]
        corr_data = corr_data[:,component,:,:] # filtering to component of time series
        corr_data = corr_data.reshape((8,8,2,392,504))[:,:,0,:,:] # reshaping and selecting correlation value index
        corr_data = ma.masked_where(broadcasted_mask==False, corr_data) # masking to land only
        if variable == 'melt':
            corr_data = ma.masked_where(broadcasted_mean_melt <= 1, corr_data) #masking based on mean melt
            
        mean_corr = np.nanmean(corr_data,axis=(2,3)) # returns the mean value across all the non-masked grid-cells, returns 8x8 array
        
        models = ['ERAI','ERA5','MetUM(044)','MetUM(011)','MAR(ERAI)','MAR(ERA5)', 'RACMO(ERAI)','RACMO(ERA5)']  
        model_numbers = np.array([0,1,2,3,4,5,6,7])
        x_index,y_index = np.meshgrid(model_numbers,model_numbers)
        values = np.flip(mean_corr,axis=0) # Note we're going to flip the y axis which is why we also flip the values
        
        plt.subplot(subplot_dimensions[0], subplot_dimensions[1], plt_number)
        
        plot = plt.scatter(x_index,y_index,s=values*scale,marker='s',c=values,vmin=vmin,vmax=1,cmap='viridis')
        plots.append(plot)
    return(plots)
        
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
            pcolormesh_basemapplot(grid_cube,basemap,vmin,vmax,cmap='Spectral')
            
def difference_to_pair_plot(input_path,basemap,vmin,vmax,variable,model_pairs,grid_cube,mask,mean_melt,row_number,subplot_dimensions):
    broadcasted_mean_melt = np.broadcast_to(mean_melt[np.newaxis,:,:],[3,392,504])
    broadcasted_mask = np.broadcast_to(mask[np.newaxis,:,:],[3,392,504])
    
    mean = np.load(f'{input_path}ensemble_stl_decomp_{variable}_mean.npy') #has shape [8,392,504]
    trend_stddev = np.load(f'{input_path}ensemble_stl_decomp_{variable}_trend_stddev.npy') #has shape [8,392,504]
    seasonal_stddev = np.load(f'{input_path}ensemble_stl_decomp_{variable}_seasonal_stddev.npy') #has shape [8,392,504]
    residual_stddev = np.load(f'{input_path}ensemble_stl_decomp_{variable}_residual_stddev.npy') #has shape [8,392,504]
    data = np.array((mean,seasonal_stddev,residual_stddev)) # data has shape [3,8,392,504]
    model1_data = data[:,model_pairs[0]]
    model2_data = data[:,model_pairs[1]]
    #ensemble_mean = data.mean(axis=1)
    data = model1_data - model2_data#np.broadcast_to(ensemble_mean[:,np.newaxis,:,:],[3,len(models),392,504])
    # data has shape [3,len(models),392,504]
    
    if variable in ['snowfall','melt']:
        ensemble_mean_trend_stddev = trend_stddev.mean(axis=0) # shape [392,504]
        data = data/np.broadcast_to(ensemble_mean_trend_stddev[np.newaxis,:,:],[3,392,504]) # turning into proportional difference
    
    if variable == 'melt':
        data = ma.masked_where(broadcasted_mask==False, data)
        data = ma.masked_where(broadcasted_mean_melt <= 1, data)
    
    for j in np.arange(0,3,1):
        plt_number = row_number*3+j+1
        plt.subplot(subplot_dimensions[0], subplot_dimensions[1], plt_number)
        grid_cube.data = data[j]
        pcolormesh_basemapplot(grid_cube,basemap,vmin,vmax,cmap='Spectral')
        
def elevation_difference_to_ensemble_plot(input_path,basemap,vmin,vmax,grid_cube):
    ensemble_elevations = np.load(f'{input_path}ensemble_elevations.npy') # has shape [6,392,504]
    mean_elevation = ensemble_elevations.mean(axis=0)
    
    counter = 1 
    for elevation in ensemble_elevations:
        difference = elevation - mean_elevation
        plt_number = counter
        plt.subplot(3, 2, plt_number)
        grid_cube.data = difference
        pcolormesh_basemapplot(grid_cube,basemap,vmin,vmax,cmap='Spectral')
        counter += 1
        
def correlation_bias_elevation_pair_plot(input_path,basemap,xlims,ylims,variable,model_pairs,mask,mean_melt,subplot_dimensions,plt_number):    
    mean = np.load(f'{input_path}ensemble_stl_decomp_{variable}_mean.npy') #has shape [8,392,504]
    trend_stddev = np.load(f'{input_path}ensemble_stl_decomp_{variable}_trend_stddev.npy') #has shape [8,392,504]
    model1_mean = mean[model_pairs[0]]
    model2_mean = mean[model_pairs[1]]
    bias = model1_mean - model2_mean # shape [392,504]
    if variable in ['snowfall','melt']:
        ensemble_mean_trend_stddev = trend_stddev.mean(axis=0) # shape [392,504]
        bias = bias/ensemble_mean_trend_stddev # turning into proportional difference
    
    elevation = np.load(f'{input_path}ensemble_elevations.npy') # has shape [6,392,504]
    if model_pairs[0] in [4,5]:
        model1_elevation = elevation[4]
    elif model_pairs[0] in [6,7]:
        model1_elevation = elevation[5]
    else:
        model1_elevation = elevation[model_pairs[0]]
    if model_pairs[1] in [4,5]:
        model2_elevation = elevation[4]
    elif model_pairs[1] in [6,7]:
        model2_elevation = elevation[5]
    else:
        model2_elevation = elevation[model_pairs[1]]
        
    elevation_difference = model1_elevation - model2_elevation
    
    elevation_difference = ma.masked_where(mask==False, elevation_difference)
    bias = ma.masked_where(mask==False, bias)
    if variable == 'melt':
        elevation_difference = ma.masked_where(mean_melt <= 1, elevation_difference)
        bias = ma.masked_where(mean_melt <= 1, bias)
    plt.subplot(subplot_dimensions[0], subplot_dimensions[1], plt_number)
    plot = plt.hexbin(elevation_difference.flatten(), bias.flatten(),extent=(xlims[0],xlims[1],ylims[0],ylims[1]),vmin=0,vmax=500,gridsize=20,linewidths=0.2)
    correlation = pearsonr(elevation_difference.flatten(), bias.flatten())
    plt.locator_params(axis="x", nbins=5)
    plt.locator_params(axis="y", nbins=5)
    plt.xlabel('Difference in Elevation / m')
    plt.ylabel('Difference in Mean Temperature / K',labelpad=-0.05)
    return(plot,correlation[0])

def correlation_table(input_path,variables,components,mask,mean_melt):
    broadcasted_mean_melt = np.broadcast_to(mean_melt[np.newaxis,:,:],[8,8,392,504])
    broadcasted_mask = np.broadcast_to(mask[np.newaxis,:,:],[8,8,392,504])
    
    for component in components:
        for variable in variables:
            corr_data = np.load(f'{input_path}ensemble_stl_decomp_corr_{variable}.npy') # has shape [64,3,2,392,504]
            corr_data = corr_data[:,component,:,:] # filtering to component of time series
            corr_data = corr_data.reshape((8,8,2,392,504))[:,:,0,:,:] # reshaping and selecting correlation value index
            corr_data = ma.masked_where(broadcasted_mask==False, corr_data)
            if variable == 'melt':
                corr_data = ma.masked_where(broadcasted_mean_melt <= 1, corr_data)
            mean_corr = np.nanmean(corr_data,axis=(2,3))
            models = ['ERAI','ERA5','MetUM(044)','MetUM(011)','MAR(ERAI)','MAR(ERA5)', 'RACMO(ERAI)','RACMO(ERA5)'] 
            df = pd.DataFrame(data=mean_corr,columns=models,index=models)
            print(f'Correlation:{variable},{component}')
            display(df) 
            
def rmsd_table(input_path,variables,mask,mean_melt):
    broadcasted_mean_melt = np.broadcast_to(mean_melt[np.newaxis,np.newaxis,:,:],[8,8,392,504])
    broadcasted_mask = np.broadcast_to(mask[np.newaxis,np.newaxis,:,:],[8,8,392,504])
    
    for variable in variables:
        rmsd = np.load(f'{input_path}ensemble_stl_decomp_rmse_{variable}.npy') # has shape [64,392,504]
        rmsd_bias_adjusted = np.load(f'{input_path}ensemble_stl_decomp_rmse_bias_adjusted_{variable}.npy') # has shape [64,392,504]
        rmsd_bias_seasonal_adjusted = np.load(f'{input_path}ensemble_stl_decomp_rmse_bias_seasonal_adjusted_{variable}.npy') # has shape [64,392,504]
        rmsd_bias_seasonal_residual_adjusted = np.load(f'{input_path}ensemble_stl_decomp_rmse_bias_seasonal_residual_adjusted_{variable}.npy') # has shape [64,392,504]

        rmsd_data = [rmsd,rmsd_bias_adjusted,rmsd_bias_seasonal_adjusted,rmsd_bias_seasonal_residual_adjusted]
        rmsd_data_mean = []
        ensemble_mean_trend_stddev = np.load(f'{input_path}ensemble_stl_decomp_{variable}_trend_stddev.npy').mean(axis=0) #has shape [392,504], will be used to express RMSD as proportion
        
        for data in rmsd_data:
            data = data.reshape((8,8,392,504)) # reshaping 
            data = ma.masked_where(broadcasted_mask==False, data)
            if variable == 'melt':
                data = ma.masked_where(broadcasted_mean_melt <= 1, data)
            if variable in ['snowfall','melt']:
                data = data/np.broadcast_to(ensemble_mean_trend_stddev[np.newaxis,np.newaxis,:,:],[8,8,392,504]) # turning into proportional difference
            rmsd_data_mean.append(np.nanmean(data,axis=(2,3))) # each array element is shape [8,8]
        
        models = ['ERAI','ERA5','MetUM(044)','MetUM(011)','MAR(ERAI)','MAR(ERA5)', 'RACMO(ERAI)','RACMO(ERA5)'] 
        tables = ['Proportional RMSD','% Reduction Trend','% Reduction Trend and Season','% Reduction Trend, Season and Residual']
        counter = 0
        for data in rmsd_data_mean:
            if counter >=1 :
                data = (rmsd_data_mean[0]-data)/rmsd_data_mean[0]*100
                df = pd.DataFrame(data=data,columns=models,index=models)
                with pd.option_context('display.float_format', '{:,.1f}%'.format):
                    print(f'RMSE:{tables[counter]},{variable}')
                    display(df)
            else:
                df = pd.DataFrame(data=data,columns=models,index=models)
                print(f'RMSE:{tables[counter]},{variable}')
                display(df)
            counter += 1

def stl_example_plot(input_path,labels_fontsize,ylabel_pad,linewidth,year_min,year_max,year_tick_freq,title_height):
    
    variables = ['snowfall','temperature','melt']
    y_labels = ["Snowfall/mmWEq","Temperature/K","Melt/mmWEq"]
    col_numbers = [1,2,3]
    ylimits = [[0,300,20,120,-60,100,-100,130],[240,275,255,267,-13,15,-9,7.5],[0,270,0,45,-40,150,-80,170]]
    ystepsizes = [[50,20,30,40],[5,2,5,3],[50,10,40,40]]
    titles = ['Original Time Series','Trend Component','Seasonal Component','Residual Component']
    
    for variable,col_number,y_label,ylimit,ystepsize in zip(variables,col_numbers,y_labels,ylimits,ystepsizes):
        larsen_c_data = np.load(f'{input_path}ensemble_stl_decomp_{variable}_larsen_c.npy') # shape [8,3,456]
        original_larsen_c_data = larsen_c_data.sum(axis=1)

        x = np.arange(0,456,1)/12+1981
        ensemble_names = ['ERAI','ERA5','MetUM(044)','MetUM(011)','MAR(ERAI)','MAR(ERA5)','RACMO(ERAI)','RACMO(ERA5)']
        for i in np.arange(0,4,1): #
            if i==0:
                data = original_larsen_c_data
            else:
                data = larsen_c_data[:,i-1]

            plt.subplot(4, 3, 3*i+col_number)
            for j in np.arange(0,8,1):
                plt.plot(x,data[j],label = ensemble_names[j],alpha=0.7, linewidth=linewidth)
            plt.plot(x,data.mean(axis=0),label = 'Ensemble Mean',alpha=0.7, linewidth=linewidth,color='k')
            plt.ylim((ylimit[i*2],ylimit[i*2+1]))
            plt.ylabel(y_label,fontsize=labels_fontsize,labelpad=ylabel_pad)
            plt.xlim([year_min,year_max])
            plt.xticks(np.arange(year_min, year_max+1, year_tick_freq),fontsize=labels_fontsize)
            plt.yticks(np.arange(ylimit[2*i], ylimit[2*i+1]+1, ystepsize[i]),fontsize=labels_fontsize)
            plt.gca().set_title(titles[i],loc='center', x=0.5, y=title_height,fontsize=labels_fontsize)
            if i==3:
                plt.xlabel("Year",fontsize=labels_fontsize)
    
def rmsd_average_table(input_path,variables,mask,mean_melt):
    broadcasted_mean_melt = np.broadcast_to(mean_melt[np.newaxis,np.newaxis,:,:],[8,8,392,504])
    broadcasted_mask = np.broadcast_to(mask[np.newaxis,np.newaxis,:,:],[8,8,392,504])
    rmsd_data_values = []
    columns = ['Snowfall','Near-Surface Air Temperature','Melt']
    rows = ['RMSD','M.Adjusted','M.&S.Adjusted','M.,S.&R. Adjusted']
    
    for variable in variables:
        rmsd = np.load(f'{input_path}ensemble_stl_decomp_rmse_{variable}.npy') # has shape [64,392,504]
        rmsd_bias_adjusted = np.load(f'{input_path}ensemble_stl_decomp_rmse_bias_adjusted_{variable}.npy') # has shape [64,392,504]
        rmsd_bias_seasonal_adjusted = np.load(f'{input_path}ensemble_stl_decomp_rmse_bias_seasonal_adjusted_{variable}.npy') # has shape [64,392,504]
        rmsd_bias_seasonal_residual_adjusted = np.load(f'{input_path}ensemble_stl_decomp_rmse_bias_seasonal_residual_adjusted_{variable}.npy') # has shape [64,392,504]

        rmsd_data = [rmsd,rmsd_bias_adjusted,rmsd_bias_seasonal_adjusted,rmsd_bias_seasonal_residual_adjusted]
        rmsd_data_average = []
        ensemble_mean_trend_stddev = np.load(f'{input_path}ensemble_stl_decomp_{variable}_trend_stddev.npy').mean(axis=0) #has shape [392,504], will be used to express RMSD as proportion
        
        for data in rmsd_data:
            data = data.reshape((8,8,392,504)) # reshaping 
            data = ma.masked_where(broadcasted_mask==False, data)
            if variable == 'melt':
                data = ma.masked_where(broadcasted_mean_melt <= 1, data)
            if variable in ['snowfall','melt']:
                data = data/np.broadcast_to(ensemble_mean_trend_stddev[np.newaxis,np.newaxis,:,:],[8,8,392,504]) # turning into proportional difference
            rmsd_data_average.append(np.nanmean(data,axis=(2,3))) # each array element is shape [8,8]
        
        rmsd_data_average = [np.nanmean(i[np.triu_indices(8, k = 1)],axis=0) for i in rmsd_data_average] # returns shape [3]
        rmsd_data_average[1:] = 1-rmsd_data_average[1:]/rmsd_data_average[0]
        rmsd_data_values.append(rmsd_data_average)                         
    
    df = pd.DataFrame(data=np.array(rmsd_data_values).T,columns=columns,index=rows)
    print(f'RMSD and Adjustments')
    display(df)
                   
def relative_strength_table(input_path,variables,mask,mean_melt):
    broadcasted_mean_melt = np.broadcast_to(mean_melt[np.newaxis,:,:],[3,392,504])
    broadcasted_mask = np.broadcast_to(mask[np.newaxis,:,:],[3,392,504])
    data = []
    columns = ['Snowfall','Near-Surface Air Temperature','Melt']
    rows = ['Trend','Seasonal','Residual']
    for variable in variables:
        trend_stddev = np.load(f'{input_path}ensemble_stl_decomp_{variable}_trend_stddev.npy').mean(axis=0) #has shape [392,504]
        seasonal_stddev = np.load(f'{input_path}ensemble_stl_decomp_{variable}_seasonal_stddev.npy').mean(axis=0) #has shape [392,504]
        residual_stddev = np.load(f'{input_path}ensemble_stl_decomp_{variable}_residual_stddev.npy').mean(axis=0) #has shape [392,504]
        total_variance = trend_stddev**2+seasonal_stddev**2+residual_stddev**2
        
        relative_strengths = np.array([trend_stddev**2,seasonal_stddev**2,residual_stddev**2])/total_variance
        relative_strengths = ma.masked_where(broadcasted_mask==False, relative_strengths)
        if variable == 'melt':
            relative_strengths = ma.masked_where(broadcasted_mean_melt <= 1, relative_strengths)
        
        data.append(relative_strengths.mean(axis=(1,2)))
    df = pd.DataFrame(data=np.array(data).T*100,columns=columns,index=rows)
    with pd.option_context('display.float_format', '{:,.0f}%'.format):
        print(f'Relative Strengths')
        display(df)
    
    
    
