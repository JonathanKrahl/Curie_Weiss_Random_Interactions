# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 19:44:30 2024

@author: jonak
"""

import os
import numpy as np

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