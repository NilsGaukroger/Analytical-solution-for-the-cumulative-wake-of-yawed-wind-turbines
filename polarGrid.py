# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 10:54:14 2022

@author: nilsg
"""

figpath = 'C:/Users/nilsg/Dropbox/Apps/Overleaf/Thesis/figures/02_cfdSetup/'

import numpy as np
import matplotlib.pyplot as plt
from post_utils import set_size

plt.rcParams.update({
  "text.usetex": True,
  "font.family": "Tex Gyre Adventor"
})
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['lines.linewidth'] = 2

textwidth = 448.0 # [pt]

#%% Discretisation
nr = 8
nt = 8

#%% Plotting
figsize = set_size(textwidth,fraction=0.75)
fig, ax = plt.subplots(1,1,figsize=(figsize[0],figsize[0]))

yr = np.empty((nr+1,nt+1))
xr = np.empty((nr+1,nt+1))

y_sp = np.empty((nr+1,nt+1))
x_sp = np.empty((nr+1,nt+1))

for ir in range(nr+1):
    for it in range(nt+1):
        yr[ir,it] = (ir/nr)*np.cos((it/nt)*2*np.pi)
        xr[ir,it] = (ir/nr)*np.sin((it/nt)*2*np.pi)
        if ir > 0:
            y_sp[ir,it] = np.mean([yr[ir-1,it-1],yr[ir,it]])
            x_sp[ir,it] = np.mean([xr[ir-1,it-1],xr[ir,it]])
        else:
            y_sp[ir,it] = 0
            x_sp[ir,it] = 0
    ax.plot(yr[ir,:],xr[ir,:],ls='-',c='k',zorder=1)
    ax.scatter(y_sp[ir,1:],x_sp[ir,1:],c='r',zorder=2)
    if ir == nr:
        ax.plot(yr[ir,:],xr[ir,:],ls='-',c='k',zorder=1,label='Shape cells')
        ax.scatter(y_sp[ir,1:],x_sp[ir,1:],c='r',zorder=2,label='Sample points')

for it in range(0,nt):
    ax.plot([0, np.cos((it/nt)*2*np.pi)],[0, np.sin((it/nt)*2*np.pi)], ls='-', c='k',zorder=1)
    
ax.set_xlabel("$y'/R$")
ax.set_ylabel("$x'/R$")
ax.set_xlim([-1.2, 1.2])
ax.set_ylim([-1.2, 1.2])
ax.invert_xaxis()
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
ax.legend(loc='center right', bbox_to_anchor=(-0.03,0.5),frameon=False)

# fig.savefig(figpath + '/polarGrid_{:d}_{:d}.pdf'.format(nr,nt), bbox_inches='tight')

#%% Full domain
# Discretisation
ny = 5
nz = 5

y = np.linspace(-1,1,ny+1)
z = np.linspace(-1,1,ny+1)

# Plotting
for iy in range(ny+1):
    ax.plot([y[iy]]*2,[-1, 1],c='b')
for iz in range(nz+1):
    ax.plot([-1,1],[z[iz]]*2,c='b')
    if iz == nz:
        ax.plot([-1,1],[z[iz]]*2,c='b',label='Domain cells')
ax.legend(loc='center right', bbox_to_anchor=(-0.02,0.5),frameon=False,markerfirst=False)

fig.savefig(figpath + '/polarGrid_{:d}_{:d}_domain.pdf'.format(nr,nt), bbox_inches='tight')
plt.show()