# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 17:13:41 2024

@author: jonak
"""

import sys
import os

# Add the parent directory to the sys.path
# This allows importing modules from the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

# Import the cw_metropolis function from the source.metropolis module
from source.metropolis import cw_metropolis

# Import the print_hamiltonian function from the source.utils module
from source.utils import print_hamiltonian

# Set the parameters for the Metropolis algorithm
pareto = True          # Boolean indicating if interactions are random (Pareto distribution)
alpha = 0.25           # Pareto distribution parameter
beta = 1e-5            # Inverse temperature coefficient
n_int = 10             # Grid width
dim = 2                # Dimension of the grid
steps = int(1e5)       # Number of Metropolis iterations
save = False           # Boolean indicating if results should be saved

# Run the Metropolis algorithm
# Returns the final spin configuration, magnetization density array, burn-in period, steps, and interaction matrix
cw, mag_d_array, burn_in, steps, Xi = cw_metropolis(pareto, alpha, beta, n_int, dim, steps, save=save)

# Calculate the Hamiltonian values for each spin using the final configuration and interaction matrix
ham = print_hamiltonian(cw, Xi, dim, n_int)
