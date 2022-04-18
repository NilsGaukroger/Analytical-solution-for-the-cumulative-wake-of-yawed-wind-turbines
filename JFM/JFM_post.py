# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 14:54:42 2022

@author: nilsg
"""

import numpy as np
import os
import xarray
import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)

mpl.style.use('default')
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams["legend.scatterpoints"] = 1
plt.rcParams["legend.numpoints"] = 1
plt.rcParams['grid.linestyle'] = ':' # Dotted gridlines
mpl.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.grid']=False
yd = dict(rotation=0,ha='right') # I couldn't find a way to customize these, so use a dict everytime..
plt.close('all')

#%% Inputs
alignment  = 'slanted'
turbines   = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
U_h        = 8
FigureSize = tuple(x/3 for x in [25.87,11.99])

#%% Parameters
## Turbine
D  = 126.0 # turbine diameter [m]
zh = 90.0  # hub height [m]

## Wind farm layout
nWT  = len(turbines)
x_AD_1 = np.empty((nWT))
y_AD = np.empty((nWT))
for i in range(nWT):
    x_AD_1[i-1] = np.floor((turbines[i]-1)/3) * 5*D
    if alignment == 'aligned':
        y_AD[i-1] = ((turbines[i]+1)%3) * 3*D
    elif alignment == 'slanted':
        y_AD[i-1] = ((turbines[i]+1)%3) * 3*D + np.floor((turbines[i]-1)/3) * 0.75*D

x_AD_2 = np.array(x_AD_1) - (x_AD_1[-1]-x_AD_1[0])/2 # PyWakeEllipSys convention
y_AD = np.array(y_AD) - (y_AD[-1]-y_AD[0])/2 # PyWakeEllipSys convention

## Wake domain
x_wd = np.array([-12*D, 20*D]) # wake domain x-limits
y_wd = np.array([-4*D, 4*D]) # wake domain y-limits

#%% Import relevant data using xarray
os.chdir(r'C:\Users\nilsg\OneDrive\Documents\EWEM\Thesis\PyWakeEllipSys\JFM') # change to correct directory

infile = alignment + '/flowdata.nc'
flowdata = xarray.open_dataset(infile)
U_zh     = flowdata.U.interp(z=zh)
U        = U_zh.where((flowdata.x >= x_wd[0]) & (flowdata.x <= x_wd[-1]) & (flowdata.y >= y_wd[0]) & (flowdata.y <= y_wd[-1]))
Unorm    = -(U.T - U_h)/U_h

#%% Analytical solution


#%% Contours of (U - U_h), normalised by the incoming velocity at hub height, U_h, at a horizontal plane at hub height

fig, ax = plt.subplots(1,1,figsize=FigureSize)

p = ax.contourf(U.x/D,U.y/D,Unorm,cmap='jet',levels=100)
ax.set_xlim(x_wd/D)
ax.set_ylim(y_wd/D)
ax.set_xlabel('$x/D$')
ax.set_ylabel('$y/D$')
ax.set_title('RANS')

divider = make_axes_locatable(ax)
cax     = divider.append_axes("right", size = "5%", pad=0.05)
cbar    = fig.colorbar(p, ax=ax, cax=cax)
cbar.set_label('$1-(U/U_h)$ [m/s]')

plt.show()
filename = 'Unorm_' + alignment
fig.savefig('fig/'+filename+'.pdf',bbox_inches='tight')
fig.savefig('fig/'+filename+'.svg',bbox_inches='tight')

#%% Vertical profiles of normalised streamwise velocity at various distances downwind of WT_15

poss   = np.asarray([4,6,8,12]) # downstream positions [D]
cols   = ['k','b','r','g']
z_zh   = np.linspace(1/9,3,27)

# Import digitised data
LES = []
for ip, pos in enumerate(poss):
    filename = alignment + '/' + str(pos) + 'D.csv'
    LES.append(pd.read_csv(filename,header=None,names=['U/Uh','delete']))
    LES[ip].insert(0,'z/zh',z_zh)
    LES[ip].drop('delete',inplace=True,axis=1)

U_prof = np.empty((len(poss)+1,27))

LES_plot  = []
RANS_plot = []
fig,ax = plt.subplots(1,1,figsize=FigureSize)

for ip, pos in enumerate(poss):
    # extract RANS profiles
    if alignment == 'aligned':
        U_prof[ip,:] = flowdata.U.interp(x=(10+pos)*D,y=0,z=z_zh*zh)
    elif alignment == 'slanted':
        U_prof[ip,:] = flowdata.U.interp(x=(10+pos)*D,y=1.5*D,z=z_zh*zh)
    
    # plot LES profiles
    l1, = ax.plot(LES[ip]['U/Uh'],LES[ip]['z/zh'],'o',c=cols[ip],markeredgecolor='k',label=str(pos) + 'D')
    LES_plot.append(l1)
    # plot RANS profiles
    l2, = ax.plot(U_prof[ip]/U_h,z_zh,'x',c=cols[ip],label=str(pos) + 'D')
    RANS_plot.append(l2)

# plot RANS inflow
U_prof[ip+1,:] = flowdata.U.interp(x=-20*D,y=0,z=z_zh*zh)
l2, = ax.plot(U_prof[ip+1]/U_h,z_zh,c='0.5',ls='--',label='inflow')
RANS_plot.append(l2)

LES_legend = plt.legend(handles=LES_plot, loc=(0.15,0.63), title='LES')
plt.gca().add_artist(LES_legend)
RANS_legend = plt.legend(handles=RANS_plot, loc = 'upper left', title='RANS')

ax.set_xlim(left=0.4)
ax.set_ylim(bottom=0)
ax.set_xlabel('$(U/U_h)$')
ax.set_ylabel('$z/z_h$')

plt.show()
filename = 'Uprof_' + alignment
fig.savefig('fig/'+filename+'.pdf',bbox_inches='tight')
fig.savefig('fig/'+filename+'.svg',bbox_inches='tight')