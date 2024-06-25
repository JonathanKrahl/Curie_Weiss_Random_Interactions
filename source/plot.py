# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 19:43:29 2024

@author: jonak
"""
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from statistics import mode
# Add the parent directory to the sys.path
# This allows importing modules from the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)
from source.utils import save_plot

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
