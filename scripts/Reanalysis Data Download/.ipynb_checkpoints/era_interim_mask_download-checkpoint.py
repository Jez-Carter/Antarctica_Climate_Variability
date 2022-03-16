#!/usr/bin/env python
from ecmwfapi import ECMWFDataServer
import numpy as np
server = ECMWFDataServer()

server.retrieve({
    "class": "ei",
    "dataset": "interim",
    "date": "1989-01-01",
    "expver": "1",
    "grid": "0.75/0.75",
    "levtype": "sfc",
    "param": "172.128",
    "step": "0",
    "stream": "oper",
    "time": "12:00:00",
    "type": "an",
    'area': [30, 150, -30, -150], # North, West, South, East. Default: global
    'rotation': '0.0/0.0',
    "target": '', # Needs filling in
})
