#!/bin/env python

import subprocess

print(subprocess.Popen(["squeue","--user=carter10"]))

process = subprocess.Popen(["squeue","--user=carter10"], stdout=subprocess.PIPE)

process.communicate()
