import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format':'grib',
        'variable': 'land_sea_mask',
        'area'          : [30, 150, -30, -150], # North, West, South, East. Default: global
        'rotation'      : '0.0/0.0',
        'year': '1979',
        'month': '01',
        'day': '01',
        'time': '00:00',
        #'grid': 'O640',
    },
    '' )# Destination needs filling in
