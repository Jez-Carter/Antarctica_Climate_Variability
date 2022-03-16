#!/usr/bin/env python
from ecmwfapi import ECMWFDataServer
import numpy as np
server = ECMWFDataServer()

variables = ['2m_temperature','snowfall','snowmelt']
params = ['167.128','144.128','45.128']

for i in np.arange(1979,2019,1):
    for variable,param in zip(variables,params):
        destination_filename = f'' # Needs filling in
        server.retrieve({
            "class": "ei",
            "dataset": "interim",
            "date": f'{i}-01-01/to/{i}-12-31',
            "expver": "1",
            "grid": "0.75/0.75",
            "levtype": "sfc",
            "param": f"{param}",
            "step": "3/6/9/12",
            "stream": "oper",
            "time": "00:00:00/12:00:00",
            "type": "fc",
            'area': [30, 150, -30, -150], # North, West, South, East. Default: global
            'rotation': '0.0/0.0',
            "target": f'' # Needs filling in,
        })
