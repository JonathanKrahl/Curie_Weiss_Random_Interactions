# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 19:46:02 2024

@author: jonak
"""

import numpy as np
from initialize import initialize_geometry
from utils import symmetrize
from plot import mag_plot



def initialize_hamiltonian(pareto, alpha, beta, cw, n_int, dim):
    """
    Initialize the Hamiltonian for the Metropolis algorithm.

    Parameters:
    pareto (bool): If True, interactions are random following a Pareto distribution, otherwise set to ones.
    alpha (float): Pareto distribution parameter.
    beta (float): Inverse temperature coefficient.
    cw (array-like): Array of spin values.
    n_int (int): Total number of spins.
    dim (int): Dimension of the grid.

    Returns:
    tuple: Interaction matrix (xi), initial Hamiltonian (ham), constant C.
    """
    
    if pareto:
        xi = np.random.default_rng().pareto(alpha, n_int**2).reshape(n_int, n_int)
        xi = symmetrize(xi)  # Ensure the interaction matrix is symmetric
    else:
        xi = np.ones((n_int, n_int))  # Uniform interaction matrix for non-Pareto case
    
    C = dim * beta / n_int  # Constant according to the CWM model with interactions
    
    # Calculate the initial Hamiltonian
    ham = -C * cw.dot(np.matmul(xi, cw))
    
    return xi, ham, C



def cw_metropolis(pareto, alpha, beta, N, dim, steps, save=False):
    """
    Perform the Metropolis algorithm for a given system configuration.

    Parameters:
    pareto (bool): Set to True if interactions should be random, otherwise False.
    alpha (float): Pareto parameter.
    beta (float): Inverse temperature coefficient.
    N (int): Grid width.
    dim (int): Dimension of grid.
    steps (int): Number of Metropolis iterations.

    Returns:
    tuple: Final spin configuration, magnetization density array, burn-in period, total steps, interaction matrix.
    """
    
# =============================================================================
# Basic definitions such as geometry, initial condition, constants
# =============================================================================
    (n_int, cw, cw_sum, n_iter, mag_density_array, 
    burn_in, mag_density_init) = initialize_geometry(N, dim, steps)
    # n_int            : Integer       - Total number of spins
    # cw               : Integer Array - Array of values of n_int spins
    # cw_sum           : Integer       - Sum of spins
    # n_iter           : Integer       - Number of iterations
    # mag_density_array: Double Array  - Records the magnetization density in every step
    # burn_in          : Integer       - Number of samples thrown away
    # mag_density_init : Double        - Initial magnetization density
    
    xi, ham, C = \
    initialize_hamiltonian(pareto, alpha, beta, cw, n_int, dim)
    # xi : 2D Double Array - Contains interaction values
    # ham: Double          - Current Hamiltonian
    # C  : Double          - Constant according to CWM-model

# =============================================================================
# Metropolis Algorithm
# =============================================================================

    # Perform Metropolis
    for i in range(n_iter):        
        # Select a random spin
        select = np.random.randint(0, n_int) 
        
        # Compute new Hamiltonian
        ham_new = ham + 4 * C * cw[select] * np.matmul(xi[select, :], cw)
        
        # Compute energy difference
        delta_E = ham_new - ham
        
        # Perform Metropolis test
        rand_u = np.random.random()
        if rand_u < min(1, np.exp(-delta_E)):  # Metropolis criterion
            ham = ham_new  # Accept new state
            cw[select] *= -1
            cw_sum += 2 * cw[select]
        
        # Record magnetization density
        mag_density_array[i] = cw_sum / n_int

# =============================================================================
# Postprocessing 
# =============================================================================
    mag_plot(mag_density_array, burn_in, steps, beta, alpha, 
             mag_density_init, pareto, save=save)
    
    return cw, mag_density_array, burn_in, steps, xi
