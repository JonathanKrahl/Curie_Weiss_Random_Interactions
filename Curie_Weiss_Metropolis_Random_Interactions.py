# -*- coding: utf-8 -*-
"""
Created on Wed May  1 11:28:32 2024

@author: jonak
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 08:59:23 2024

Curie-Weiss-Modell

@author: jonak
"""
import numpy as np
import matplotlib.pyplot as plt
from statistics import mode
import os

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


def mag_plot(mag_d_array, burn_in, steps, beta, alpha, mag_d_init, pareto, save=False, average=False):
    """
    Plot and optionally save histograms and time series of magnetization density.

    Parameters:
    mag_d_array (ndarray): Array of magnetization density values.
    burn_in (int): Number of initial steps to discard.
    steps (int): Total number of steps in the simulation.
    beta (float): Inverse temperature coefficient.
    alpha (float): Pareto distribution parameter.
    mag_d_init (float): Initial magnetization density.
    save (bool): If True, save the plots to disk. Default is False.
    average (bool): If True, indicate that the plot represents averaged data over multiple simulations. Default is False.
    """
    
    # Create histogram plot of magnetization density
    if len(mag_d_array[mag_d_array > 0]) > 0 and len(mag_d_array[mag_d_array < 0]) > 0:
        positive_peak = mode(mag_d_array[mag_d_array >= 0])
        negative_peak = mode(mag_d_array[mag_d_array <= 0])
        label = (f"Positive Peak: {positive_peak:.2f}\n"
                 f"Negative Peak: {negative_peak:.2f}\n"
                 f"Initial MagD: {mag_d_init:.2f}")
    else:
        peak = mode(mag_d_array)
        label = (f"Peak: {peak:.2f}\n"
                 f"Initial MagD: {mag_d_init:.2f}")
    
    plt.hist(mag_d_array[burn_in:], bins=200, label=label)
    plt.xlabel('Magnetization Density')
    plt.ylabel('Occurrences')
    plt.legend()
    
    # Set title according to interactions
    mid_title = f'Pareto({alpha}) ' if pareto else 'Standard '
    title_prefix = 'Average Distribution' if average else 'Curie-Weiss-Model - Distribution'
    plt.title(f'{title_prefix} of Magnetization Density \n' + mid_title + f'Interactions, β = {beta}, steps = {steps:.0e}')
    
    if save:
        save_plot(steps, alpha, beta, "MAGDD", plt)
        
    plt.show()
    
    # Plot magnetization density over steps
    plt.plot(mag_d_array[burn_in:])
    plt.xlabel('Step')
    plt.ylabel('Magnetization Density')
    plt.title('Curie-Weiss-Model - Magnetization Density over Steps \n' + mid_title + f'Interactions, β = {beta}, steps = {steps:.0e}')
    
    if save:
        save_plot(steps, alpha, beta, "MAGDS", plt)
        
    plt.show()

def save_plot(steps, alpha, beta, plot_type, plt):
    """
    Save the current plot to a file in a structured directory format.

    Parameters:
    steps (int): Total number of steps in the simulation.
    alpha (float): Pareto distribution parameter.
    beta (float): Inverse temperature coefficient.
    plot_type (str): Type of plot ('MAGDD' or 'MAGDS').
    plt (object): Matplotlib pyplot object.
    """
    folder_name = "Plots1"
    os.makedirs(folder_name, exist_ok=True)
    
    step_folder = os.path.join(folder_name, f"Steps = {steps}")
    os.makedirs(step_folder, exist_ok=True)
    
    subfolder_name = os.path.join(step_folder, plot_type)
    os.makedirs(subfolder_name, exist_ok=True)
    
    subsubfolder_name = os.path.join(subfolder_name, f"Pareto({alpha})")
    os.makedirs(subsubfolder_name, exist_ok=True)
    
    file_path = os.path.join(subsubfolder_name, f"beta={beta}.png")
    plt.savefig(file_path, dpi=500, bbox_inches='tight')

        

def symmetrize(matrix):
    """
    Symmetrize a given square matrix.

    This function creates a symmetric matrix by mirroring the upper triangle 
    onto the lower triangle.

    Parameters:
    matrix (ndarray): The input square matrix to be symmetrized.

    Returns:
    ndarray: The symmetrized square matrix.
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square")
    
    # Symmetrize the matrix by mirroring the upper triangle onto the lower triangle
    sym_mat = matrix - np.tril(matrix, -1) + np.triu(matrix, 1).T
    
    return sym_mat


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

    
def cw_metropolis(pareto, alpha, beta, N, dim, steps):
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
             mag_density_init, pareto, save=False)
    
    return cw, mag_density_array, burn_in, steps, xi




    





