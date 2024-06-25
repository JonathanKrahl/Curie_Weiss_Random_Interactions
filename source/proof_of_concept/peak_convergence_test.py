# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 19:18:01 2024

@author: jonak

This code is a proof of concept for the convergence of peaks
It runs the Metropolis algorithm for different grid widths and prints the mode
of the magnetization density array for each grid width
"""

import sys
import os
import numpy as np
from statistics import mode

# Add the parent directory to the sys.path
# This allows importing modules from the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

# Import the cw_metropolis function from the source.metropolis module
from source.metropolis import cw_metropolis

# Loop through a range of values (10 to 90 in steps of 10) for grid width (i)
for i in np.arange(10, 100, 10):
    # Run the Metropolis algorithm with the given parameters
    # Parameters: 
    # False (no Pareto distribution, uniform interactions)
    # 0 (alpha, not used here since Pareto is False)
    # 0.3 (beta, inverse temperature coefficient)
    # i (grid width)
    # 2 (dimension of the grid)
    # 1000000 (number of Metropolis iterations)
    cw, mag_d_array, burn_in, steps, xi = cw_metropolis(False, 0, 0.3, i, 2, 1000000)
    
    # Print the mode of the magnetization density array truncated to 7 characters
    print(str(mode(mag_d_array))[:7])
