# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 09:49:00 2022

@author: nilsg
"""

import os
os.chdir('C:/Users/nilsg/OneDrive/Documents/EWEM/Thesis/NUMERICAL/PyWakeEllipSys/')
from post_utils import windTurbine, windFarm, flowcase

NREL5MW = windTurbine(D=126, zh=90, TSR=7.5)

wf = windFarm(Uinf=8, ti=0.04, x_wt=[0], y_wt=[0], wts=[NREL5MW], yaws=[40], CTs=[0.8], wrs=[True])

data_path = r'C:/Users/nilsg/OneDrive/Documents/EWEM/Thesis/NUMERICAL/PyWakeEllipSys/lss/data/'

fc = flowcase(infile = data_path + 'yaw/40/flowdata.nc', wf=wf)

#%%

u_d = fc.vel_disc('U', 1, plot=True, xlim=(-0.7,0.7))
print(u_d)

# #%% Setup

# ## Define constants
# rho = 1.225 # air density [kg/m^3]
# zh  = 90.0  # hub height [m]
# D   = 126.0 # diameter [m]

# ## Define turbine x and y coordinates and number of turbines
# x_wt = np.asarray([0, 0, 0, 5, 5, 5, 10, 10, 10, 15, 15, 15, 20, 20, 20])
# y_wt = np.asarray([6, 0, 3, 6, 0, 3, 6, 0, 3, 6, 0, 3, 6, 0, 3])
# n_wt = len(x_wt)

# ## Define z coordinates of turbine hubs
# z_wt = zh*np.ones((n_wt))

# ## Sort turbine coordinates by increasing x-coordinate
# idx  = np.argsort(x_wt)
# x_wt = x_wt[idx]
# y_wt = y_wt[idx]
# z_wt = z_wt[idx]

# ## Inflow
# Uh = 8    # hub height inflow velocity [m/s]
# TI = 0.04 # inflow turbulence intensity [-]

# #%% Analytical solution
# u_d = np.zeros((n_wt))
# for n in range(n_wt):
#     # u_d[n] = vel_disc(u, x, y, z, x_wt[n], y_wt[n], z_wt[n], D, yaw)
    
    
# #%% Functions
# def vel_disc(u, x, y, z, x_t, y_t, z_t, D, yaw):
#     ## Initialise totals
#     u_d = 0 # total contribution of all sampled points
#     n   = 0 # total number of sampled points
    
#     # For all points in yz-plane at x == x_t
#     for j in range(len(y)):
#         for k in range(len(z)):
#             in_disc = ((y[j] - y_t)**2 + (z[k] - z_t)**2) <= (D/2)**2
#             if in_disc:
#                 u_d = u_d + np.interp(x_t, x, u[:,j,k])
#                 n   = n + 1 # add to point counter
    
#     return u_d / n # average disc velocity

# def NREL5MW(u_d):
#     CT = []