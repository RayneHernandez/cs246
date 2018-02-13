#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 11:31:18 2018

@author: raynehernandez
"""

import numpy as np
import scipy as sp
from scipy import linalg

# Taking SVD of M
M = np.matrix([[1,2], [2,1], [3,4], [4,3]])

U, s, Vt = sp.linalg.svd(M)
print(U)
print(s)
print(Vt)

# Taking SVD of MtM

Mt = np.transpose(M)

evals, evecs = sp.linalg.eig(Mt.dot(M))
print(evals)
print(evecs)