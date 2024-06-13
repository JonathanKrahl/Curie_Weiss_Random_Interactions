# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 19:18:01 2024

@author: jonak
"""

import numpy as np

from Curie_Weiss_Metropolis_Random_Interactions import *

for i in np.arange(10,11):
    cw, mag_d_array, burn_in, steps, xi = cw_metropolis(False, 0, 0.3, i, 2, 1000000)
    print(str(mode(mag_d_array))[:7])