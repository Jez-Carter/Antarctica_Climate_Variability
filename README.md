# Antarctica_Climate_Variability

This code examines variability in an ensemble of RCM and Renalaysis outputs over Antarctica from ECMWF (https://www.ecmwf.int/) and the Antarctic CORDEX project (https://climate-cryosphere.org/antarctic/). The models used include:

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

A paper has been written summarising the main findings from the work, which is currently in review, see: 'Variability in Antarctic Surface Climatology Across Regional Climate Models and Reanalysis Datasets' J.Carter et al.

All plots and tables from the paper can be generated in the 'Paper Figures and Tables' and 'Appendix Figures and Tables' notebook files within the scripts folder.

The order dependency of the code to go from the raw data to the data used in the figures and tables is shown below:

![plot](./Code_structure.jpg)


