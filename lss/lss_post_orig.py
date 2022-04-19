# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 11:01:54 2022

@author: nilsg
"""

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

os.chdir('C:/Users/nilsg/OneDrive/Documents/EWEM/Thesis/PyWakeEllipSys/lss/')

import lss_utils as u
from lss_utils import flowcase

mpl.style.use('default')
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams["legend.scatterpoints"] = 1
plt.rcParams["legend.numpoints"] = 1
plt.rcParams['grid.linestyle'] = ':' # Dotted gridlines
mpl.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.grid']=True
yd = dict(rotation=0,ha='right') # I couldn't find a way to customize these, so use a dict everytime..
plt.close('all')

FigureSize = tuple(x/3 for x in [25.87,11.99])

#%% Inputs
poss = [5,6,7,8,9,10,12,14,16,18]  # downstream profile sampling points [diameters]
yaws = [0,10,20,30,40] # yaw angles [deg]
cts  = [0.2,0.4,0.6,0.8] # thrust coefficients [-]
tis  = [0.06,0.08,0.12,0.14] # turbulence intensities [-]
wwm  = 'integral' # wake width calculation method

if yaws[0] == 'all':
    yaws = [-40,-30,-20,-10,0,10,20,30,40]

#%% Parameters
# Freestream velocity
Uinf = 8 # [m/s]

## Turbine
D  = 126.0 # turbine diameter [m]
zh = 90.0  # hub height [m]

## Wind farm layout
x_wt = [0]
y_wt = [0]

## Wake domain limits
wd = [(4,30),(20,20)]
if wd[0][1] < max(poss):
    wd_list = list(wd[0])
    wd_list[1] = max(poss)+1
    wd[0] = tuple(wd_list)

#%% Create instances for varying yaw angle, CT, and TI
lss_yaws = []
for iy, yaw in enumerate(yaws):
    lss_yaws.append(flowcase(Uinf, D, zh, x_wt, y_wt, 8, wd))
    infile = 'C:/Users/nilsg/OneDrive/Documents/EWEM/Thesis/PyWakeEllipSys/lss/data/yaw/' + str(yaw) + '/flowdata.nc'
    lss_yaws[iy].load_flowdata(infile)
    
lss_cts = []
for ict, ct in enumerate(cts):
    lss_cts.append(flowcase(Uinf, D, zh, x_wt, y_wt, 8, wd))
    infile = 'C:/Users/nilsg/OneDrive/Documents/EWEM/Thesis/PyWakeEllipSys/lss/data/ct/' + str(ct) + '/flowdata.nc'
    lss_cts[ict].load_flowdata(infile)
    
lss_tis = []
for iti, ti in enumerate(tis):
    lss_tis.append(flowcase(Uinf, D, zh, x_wt, y_wt, 8, wd))
    infile = 'C:/Users/nilsg/OneDrive/Documents/EWEM/Thesis/PyWakeEllipSys/lss/data/ti/%0.2f/flowdata.nc' % ti
    lss_tis[iti].load_flowdata(infile)
    
lss_nwr = []
for iy, yaw in enumerate(yaws):
    lss_nwr.append(flowcase(Uinf, D, zh, x_wt, y_wt, 8, wd))
    infile = 'C:/Users/nilsg/OneDrive/Documents/EWEM/Thesis/PyWakeEllipSys/lss/data/wr/' + str(yaw) + '/flowdata.nc'
    lss_nwr[iy].load_flowdata(infile)
    
# lss_yaws_32cD = [flowcase(Uinf, D, zh, x_wt, y_wt, 32, wd)]
# infile = 'C:/Users/nilsg/OneDrive/Documents/EWEM/Thesis/PyWakeEllipSys/lss/data/yaw/32cD/40/flowdata.nc'
# lss_yaws_32cD[0].load_flowdata(infile)

#%% Plot 1 - effect of ... on V velocity profile
u.plot_velProf1(lss_tis, tis, 'ti', pos=5, FigureSize=FigureSize)

#%% Plot 2 - Plot plane of V at hub height for single case with profiles at several downstream positions
for iy, yaw in enumerate(yaws):
    lss_yaws[iy].plot_velProf2(yaw, poss, FigureSize=FigureSize)
    
#%% Plot 3 - effect of ... on self similarity of V velocity profile
u.plot_LSS(lss_tis, tis, 'ti', poss, wwm, xlim=(-5,5), ylim=(0,None), FigureSize=FigureSize)

#%% Plot 4 - Gaussian fit of self-similar profile for one yaw angle
sig = lss_yaws[-1].plot_LSS_Gaussian(poss, wwm, xlim=(-5,5), FigureSize=FigureSize)

#%% Plot 5 - effect of wake rotation on V velocity profile for various yaw angles
u.plot_velProf_wr(lss_yaws, lss_nwr, yaws, 'yaw', pos=5, xlim=(-5,5), FigureSize=FigureSize)

#%% Plot 6 - effect of wake rotation on self-similarity for various yaw angles
u.plot_LSS_wr(lss_yaws, lss_nwr, yaws, 'yaw', poss, wwm, xlim=(-5,5), ylim=(None,None), FigureSize=FigureSize)

#%% Plot 7 - wake centres and edges for a single yaw case
x_Ds = np.linspace(2.5,25,100)
wwm  = 'integral'
# plot_yaws = [10,20,30,40]
plot_yaws = [40]
for yaw in plot_yaws:
    idx  = [i for i in range(len(yaws)) if yaws[i] == yaw]
    lss_yaws[idx[0]].plot_wakeCentresandEdges(x_Ds, wwm, yaw, xlim=(None,np.max(x_Ds)), ylim=(-3,3), FigureSize=FigureSize)

#%% Debugging
idx = -1
for pos in poss:
    plt.plot(lss_yaws[idx].V_zh.interp(x = lss_yaws[idx].x_AD[0] + pos*lss_yaws[idx].D).y/lss_yaws[idx].D,lss_yaws[idx].V_zh.interp(x = lss_yaws[idx].x_AD[0] + pos*lss_yaws[idx].D))
plt.xlim((-5,5))