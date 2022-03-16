# The following script was used to download ERA5 Data through an API connection.

import cdsapi
import numpy as np

c = cdsapi.Client()

variables = ['2m_temperature','snowfall','snowmelt']

for i in np.arange(1979,2021,1):
    for variable in variables:
        destination_filename = f'' # Needs filling in

            c.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'format': 'grib',
                    'variable': variable,
                    'year': f'{i}',
                    'month': [
                        '01', '02', '03',
                        '04', '05', '06',
                        '07', '08', '09',
                        '10', '11', '12',
                    ],
                    'day': [
                        '01', '02', '03',
                        '04', '05', '06',
                        '07', '08', '09',
                        '10', '11', '12',
                        '13', '14', '15',
                        '16', '17', '18',
                        '19', '20', '21',
                        '22', '23', '24',
                        '25', '26', '27',
                        '28', '29', '30',
                        '31',
                    ],
                    'time': [
                        '00:00', '01:00', '02:00',
                        '03:00', '04:00', '05:00',
                        '06:00', '07:00', '08:00',
                        '09:00', '10:00', '11:00',
                        '12:00', '13:00', '14:00',
                        '15:00', '16:00', '17:00',
                        '18:00', '19:00', '20:00',
                        '21:00', '22:00', '23:00',
                    ],
                    'area'          : [30, 150, -30, -150], # North, West, South, East. Default: global (Note these are the coordinates that work with a rotation of [0,0])
                    'rotation'      : '0.0/0.0',
                    'format'        : 'grib' # Supported format: grib and netcdf. Default: grib
                },
                destination_filename)
