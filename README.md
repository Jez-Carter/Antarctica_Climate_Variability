# Antarctica_Climate_Variability

This code examines variability in an ensemble of RCM and Renalaysis outputs over Antarctica:

| **RCM/Reanalysis Dataset** | **Domain** | **Driving Data** | **H.Resolution [km]** | **Label**   |
|----------------------------|------------|------------------|-----------------------|-------------|
| ERA-Interim                | Global     | -                | 79                    | ERAI        |
| ERA5                       | Global     | -                | 31                    | ERA5        |
| MetUMv11.1                 | Antarctica | ERA-Interim      | 12                    | MetUM(011)  |
| MetUMv11.1                 | Antarctica | ERA-Interim      | 50                    | MetUM(044)  |
| MARv3.10                   | Antarctica | ERA-Interim      | 35                    | MAR(ERAI)   |
| MARv3.10                   | Antarctica | ERA5             | 35                    | MAR(ERA5)   |
| RACMOv2.3p2                | Antarctica | ERA-Interim      | 27                    | RACMO(ERAI) |
| RACMOv2.3p2                | Antarctica | ERA5             | 27                    | RACMO(ERA5) |

The main scripts for the project are stored within the /scripts folder, which contains a /Jasmin_Batch_Scripts folder as most of the code was run using the JASMIN supercomputer and with batch script submissions. 

In addition to python scripts there are jupyter notebook files within the /scripts folder. These notebooks provide some additional detail on what the .py files are doing and the inputs they take and outputs they produce. 

There are 3 modules within the /src folder: helper_functions; loader_functions; ploter_functions. These modules contain useful functions that are called from the different .py script files. 

A paper has been written summarising the main findings from the work: .............

All plots and tables from the paper can be generated in the 'Paper Figures and Tables' and 'Appendix Figures and Tables' notebook files within the scripts folder.

To order the code needs to be run in to execute from start-to-finish is:

- Run 'importing_and_adjusting.py', this code loads the raw data using iris and adds 2D latitude,longitude auxilliary coordinates as well as aggregating the data to monthly timestamps, see 'Examining Imported Data.ipynb' for more information.
- Run 'regridding_imported_data.py', this code loads regrids the imported data from all models onto the MetUM(011) model grid, see 'Examining Regridded Cubes.ipynb' for more information.
- Run 'creating_numpy_ensemble_arrays.py', this code creates large (5GB) [8,456,392,504] numpy arrays for each variable of interest, see 'Examining Ensemble Numpy Arrays.ipynb' for more information. [NOTE I NEED TO EDIT NOTEBOOK]
- Run 'stl_decomposition_on_numpy_ensemble_arrays.py', this code takes the large (5GB) numpy arrays above and decomposes the time series into a trend, seasonal and residual component, resulting in 15GB numpy array files of shape [8,3,456,392,504], see 'Examining STL Decomposition Data.ipynb' for more information. [NOTE I NEED TO EDIT NOTEBOOK]
- Run 'creating_ensemble_component_average_arrays.py', this code takes the 15GB decomposed data and calculates the mean of the trend component as well as the standard deviation in the trend, seasonal and residual components. This produces arrays of shape [8,392,504].
- Run 'creating_mean_melt_array.py', this code takes the 15GB decomposed data and calculates the mean monthly melt across the ensemble. This produces an array of shape [392,504].
- Run 'creating_correlation_ensemble_arrays.py', this code takes the 15GB decomposed data and calculates the correlation (and corresponding p value) between every output pair for each time series component. This produces an array of shape [64,3,2,392,504].
- Run 'creating_rmse_ensemble_arrays.py', this code takes the 15GB decomposed data and calculates the RMSD between every output pair. This is done for the original data and also for the data after removing differences in the mean, seasonal standard deviations and residual standard deviations. This produces arrays of shape [64,392,504].
- Run 'creating_elevation_ensemble_array.py', this code takes elevation data from each model, regrids it onto the MetUM(011) grid and outputs a numpy array containing the values. This produces an array of shape [6,392,504] (Note both MAR and RACMO simulations have the same DEM profiles, hence the shape 6 rather than 8).
- Run 'sampling_stl_decomp_larsen_c.py', this code takes the 15GB decomposed data and filters it to a single grid-cell over the Larsen C ice shelf. This produces arrays of shape [8,3,456].

