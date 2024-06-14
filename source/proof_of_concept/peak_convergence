# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 19:18:01 2024

@author: jonak
"""

import sys
import os
import numpy as np
from statistics import mode

# Add the parent directory to the sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

from source.metropolis import cw_metropolis

for i in np.arange(10,100,10):
    cw, mag_d_array, burn_in, steps, xi = cw_metropolis(False, 0, 0.3, i, 2, 1000000)
    print(str(mode(mag_d_array))[:7])
