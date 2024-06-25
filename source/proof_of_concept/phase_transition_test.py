
"""
Created on Fri Jun 14 08:54:02 2024

@author: jonak

Proof of concept for the Metropolis Algorithm.
A bimodal distribution only occurs for beta > 0.25
"""
import sys
import os
import numpy as np

# Add the parent directory to the sys.path
# This allows importing modules from the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

# Import the cw_metropolis function from the source.metropolis module
from source.metropolis import cw_metropolis

# Loop through a range of values (10 to 100 in steps of 10) for grid width (i)
for i in np.arange(10, 110, 10):
    # Run the Metropolis algorithm with the given parameters
    # Parameters: 
    # False (no Pareto distribution, uniform interactions)
    # 0 (alpha, not used here since Pareto is False)
    # 0.251/0.249 (beta, inverse temperature coefficient)
    # i (grid width)
    # 2 (dimension of the grid)
    # 1000000 (number of Metropolis iterations)
    cw, mag_d_array, burn_in, steps, xi = cw_metropolis(False, 0, 0.251, i, 2, 1000000)
    cw, mag_d_array, burn_in, steps, xi = cw_metropolis(False, 0, 0.249, i, 2, 1000000)
