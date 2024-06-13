# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 19:37:11 2024

@author: jonak
"""
import numpy as np

def initialize_geometry(N, dim, steps):
    """
    Initialize the geometry and parameters for the Metropolis algorithm.

    Parameters:
    N (int): Grid width.
    dim (int): Dimension of the grid.
    steps (int): Number of Metropolis iterations.

    Returns:
    tuple: 
        n_int (int): Total number of spins.
        cw (ndarray): Array of spin values.
        cw_sum (int): Sum of spin values.
        n_iter (int): Number of iterations.
        mag_d_array (ndarray): Array to record magnetization density at each step.
        burn_in (int): Number of iterations to discard as burn-in.
        mag_d_init (float): Initial magnetization density.
    """
    
    n_int = N**dim  # Total number of spins
    rng = np.random.default_rng()
    cw = rng.choice([-1, 1], size=n_int)  # Initialize spins randomly to -1 or 1
    cw_sum = np.sum(cw)  # Sum of spin values
    mag_d_init = cw_sum / n_int  # Initial magnetization density
    n_iter = steps  # Number of Metropolis iterations
    mag_d_array = np.zeros(n_iter)  # Array to store magnetization density
    burn_in = int(0.1 * n_iter)  # Number of samples to discard as burn-in
    
    return n_int, cw, cw_sum, n_iter, mag_d_array, burn_in, mag_d_init