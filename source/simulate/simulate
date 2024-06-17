# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 17:13:41 2024

@author: jonak
"""

import sys
import os
import numpy as np
# Add the parent directory to the sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

from source.metropolis import cw_metropolis

cw, mag_d_array, burn_in, steps, xi = cw_metropolis(True, 1.25, 0.12, 10, 2, 1000000, save=True)
