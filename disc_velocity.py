# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 11:43:47 2022

@author: nilsg
"""

#%% Module imports
import numpy as np
from copy import copy
import xarray as xr
import matplotlib.pyplot as plt

#%% Setup
# Layout alignment
alignment = 'slanted' # 'aligned' or 'slanted'

# Choice of method
method = "original" # "original" or "modified" (see Bastankhah et al., 2020)

# Define constants
rho = 1.225 # air density [kg/m^3]
zh  = 90.0  # turbine hub height [m]
D   = 126.0 # turbine rotor diameter [m]

# Define number of turbines
n_t = 15

# Define layout parameters
s_x     = 7    # streamwise inter-turbine spacing [D]
s_y     = 4    # lateral inter-turbine spacing [D]
stagger = 0.75 # stagger [D]

# Define turbine x and y positions
x_t = np.empty((n_t))
y_t = np.empty((n_t))
for i_t in range(n_t):
    x_t[i_t] = np.floor(i_t/3) * s_x
    if alignment == 'aligned':
        y_t[i_t] = ((i_t-1)%3) * s_y - s_y
    elif alignment == 'slanted':
        y_t[i_t] = ((i_t-1)%3) * s_y + np.floor(i_t/3) * stagger - s_y
    else:
        print('Incorrect alignment specification.')
        break

# Define turbine z positions
z_t = zh * np.ones((n_t))

# Define yaw angles
yaws = np.asarray([25]*n_t)

# Sort turbine x, y, z positions by increasing x
idx = np.argsort(x_t)
x_t = x_t[idx]
y_t = y_t[idx]
z_t = z_t[idx]
yaws = yaws[idx]

print('Setup complete.')

#%% Flow domain
x = np.linspace(min(x_t)-2, max(x_t)+10,  100)*D
y = np.linspace(min(y_t)-1,  max(y_t)+1,  100)*D
z = np.linspace(         0,           2,  100)*zh

nx, ny, nz = len(x), len(y), len(z)

#%% Inflow
# Define inflow parameters
U_h = 8    # streamwise hub height inflow velocity [m/s]
V_h = 0    # lateral hub height inflow velocity [m/s]
I0  = 0.10 # hub height total inflow turbulence intensity [-]

# Preallocate analytical velocity field
U0 = np.zeros((nx, ny, nz)) # streamwise velocity
V0 = np.ones((nx, ny, nz))*-0.5 # lateral velocity

# Preallocate streamwise velocity with adiabatic log law inflow
u_s   = 0.5  # friction velocity [m/s]
kappa = 0.4  # von Kármán constant [-]
z0    = 0.15 # roughness length [m]

U_in = np.zeros((nz))
U_in[1:] = (u_s/kappa) * np.log(z[1:]/z0)

for i in range(nx):
    U0[i,:,:] = U_in * x[i]/8

print('Inflow preallocated.')

#%% Stuff
n = 1

x_t = x_t[n]
y_t = y_t[n]
z_t = z_t[n]
yaw = np.deg2rad(yaws[n])

# Initialise velocity fields
U = copy(U0)
V = copy(V0)

# Make velocities into xarray DataArray
flowdata = xr.Dataset(
    data_vars = dict(
        U  = (["x","y","z"], U),
        V  = (["x","y","z"], V),
        U0 = (["x","y","z"], U0),
        V0 = (["x","y","z"], V0)
    ),
    coords = dict(
        x = ("x", x),
        y = ("y", y),
        z = ("z", z)),
    attrs = dict(description = "Analytical solution"),
)

#%% vel_disc()
n = 128

# Find edges of rotor in x, y, z
lower = (x_t - (D/2)*np.sin(yaw),
         y_t - (D/2)*np.cos(yaw),
         z_t - (D/2)
         )
upper = (x_t + (D/2)*np.sin(yaw),
         y_t + (D/2)*np.cos(yaw), 
         z_t + (D/2)
         )

# Rotated rotor coordinates
yp_rotor = np.linspace(y_t - D/2, y_t + D/2, n)

# Discretise rotor surface as rectangle
x_rotor = xr.DataArray(np.linspace(lower[0], upper[0], n),
                           dims="y'", coords={"y'": yp_rotor})
y_rotor = xr.DataArray(np.linspace(lower[1], upper[1], n),
                           dims="y'", coords={"y'": yp_rotor})
z_rotor = xr.DataArray(np.linspace(lower[2], upper[2], n),
                           dims="z")

# 2D interpolation at hub height
flowdata_rec = flowdata.interp(x=x_rotor, y=y_rotor, z=z_rotor)

fig, ax = plt.subplots(1,1)
p = ax.contourf(flowdata_rec["y'"]/D, flowdata_rec.z/D, flowdata_rec.U.T/U_h)
ax.set_xlabel("$y'/D$")
ax.set_ylabel('$z/D$')

# Add colourbar
plt.subplots_adjust(right=0.8,
                    hspace=0.3)
cax = fig.add_axes([0.825, 0.15, 0.03, 0.7])
cbar = fig.colorbar(p, cax=cax)
cbar.set_label('$U/U_h$')

flowdata_rotor = flowdata_rec.where(np.sqrt((flowdata_rec["y'"] - y_t)**2 + (flowdata_rec.z - z_t)**2) < (D/2),
                         other=0.0)

fig, ax = plt.subplots(1,1)
ax.contourf(flowdata_rotor["y'"]/D, flowdata_rotor.z/zh, flowdata_rotor.U.T/U_h)
ax.set_xlabel("$y'/D$")
ax.set_ylabel('$z/z_h$')

# Add colourbar
plt.subplots_adjust(right=0.8,
                    hspace=0.3)
cax = fig.add_axes([0.825, 0.15, 0.03, 0.7])
cbar = fig.colorbar(p, cax=cax)
cbar.set_label('$U/U_h$')