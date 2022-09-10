# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 19:12:25 2022

@author: nilsg
"""

#%% Module imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.transforms as mtransforms
from post_utils import windTurbine, windFarm, flowcase, set_size, draw_AD, wakeCentre
from copy import copy

# Reset to defaults
plt.rcParams.update(plt.rcParamsDefault)

# Edit the font, font size, and axes width
plt.rcParams.update({
  "text.usetex": True,
  "font.family": "Tex Gyre Adventor"
})
plt.rcParams['font.size'] = 10
plt.rcParams['lines.linewidth'] = 1.5

data_path = 'C:/Users/nilsg/OneDrive/Documents/EWEM/Thesis/NUMERICAL/PyWakeEllipSys/flowdata/'

fig_path = 'C:/Users/nilsg/Dropbox/Apps/Overleaf/Thesis/figures/06_modelPredictions/'

textwidth = 448.0 # [pts]

#%% Wind turbine & wind farm objects

NREL5MW = windTurbine(D=126.0, zh=90.0, TSR=7.5)

wf_both = windFarm(Uinf=8,
              ti=0.06,
              x_wt=[0,7],
              y_wt=[0,0],
              wts=[NREL5MW]*2,
              yaws=[25,0],
              CTs=[0.8]*2,
              wrs=[True]*2)

wf_upstreamOnly = windFarm(Uinf=8,
              ti=0.06,
              x_wt=[0],
              y_wt=[0],
              wts=[NREL5MW],
              yaws=[25],
              CTs=[0.8],
              wrs=[True])

wf_downstreamOnly = windFarm(Uinf=8,
              ti=0.06,
              x_wt=[7],
              y_wt=[0],
              wts=[NREL5MW],
              yaws=[0],
              CTs=[0.8],
              wrs=[True])

#%% Import flowcases
both = flowcase(dir_path = data_path + 'SWS/both/',
                wf = copy(wf_both),
                cD = 8)

downstream = flowcase(dir_path = data_path + 'SWS/downstreamOnly/',
                      wf = copy(wf_downstreamOnly),
                      cD = 8)

upstream = flowcase(dir_path = data_path + 'SWS/upstreamOnly/',
                    wf = copy(wf_upstreamOnly),
                    cD = 8)

#%% Plot V_zh for each flowcase to check coords
# Create figure and axes objects
fig, axs = plt.subplots(3,1,
                       figsize=set_size(textwidth,
                                        1),
                       sharex=True,
                       sharey=True)

# Axes limits
xlims = (min(both.wf.x_AD)-2, max(both.wf.x_AD)+8)
ylims = (-2, 2)

## Plot contours
# Both
p = axs[0].contourf(both.flowdata.x/both.wf.D,
                    both.flowdata.y/both.wf.D,
                    both.V_zh.T/both.wf.Uinf,
                    cmap='jet')

# Upstream
p = axs[1].contourf(upstream.flowdata.x/upstream.wf.D,
                    upstream.flowdata.y/upstream.wf.D,
                    upstream.V_zh.T/upstream.wf.Uinf,
                    cmap='jet')

# Downstream
p = axs[2].contourf(downstream.flowdata.x/downstream.wf.D,
                    downstream.flowdata.y/downstream.wf.D,
                    downstream.V_zh.T/downstream.wf.Uinf,
                    cmap='jet')

axs[-1].set_xlim(xlims)
axs[-1].set_ylim(ylims)

axs[-1].set_xlabel('$x/D$')
for ax in axs:
    ax.set_ylabel('$y/D$')

plt.show()

#%% Shift flowdata to be in correct places
both.flowdata = both.flowdata.shift(x = int(3.5*both.cD))

downstream.flowdata = downstream.flowdata.shift(x = int(7*downstream.cD))

both.V_zh = both.flowdata.V.interp(z=both.wf.zh)

downstream.V_zh = downstream.flowdata.V.interp(z=downstream.wf.zh)

#%% Plot V_zh for each flowcase to check coords
# Create figure and axes objects

# Axes limits
xlims = (-2, max(both.wf.x_wt)+8)
ylims = (-2, 2)

fig, axs = plt.subplots(3,1,
                       figsize=set_size(textwidth,
                                        1),
                       sharex=True,
                       sharey=True)

# Axes limits
xlims = (min(both.wf.x_AD)-2, max(both.wf.x_AD)+8)
ylims = (-2, 2)

## Plot contours
# Both
p = axs[0].contourf(both.flowdata.x/both.wf.D,
                    both.flowdata.y/both.wf.D,
                    both.V_zh.T/both.wf.Uinf,
                    cmap='jet')

# Upstream
p = axs[1].contourf(upstream.flowdata.x/upstream.wf.D,
                    upstream.flowdata.y/upstream.wf.D,
                    upstream.V_zh.T/upstream.wf.Uinf,
                    cmap='jet')

# Downstream
p = axs[2].contourf(downstream.flowdata.x/downstream.wf.D,
                    downstream.flowdata.y/downstream.wf.D,
                    downstream.V_zh.T/downstream.wf.Uinf,
                    cmap='jet')

axs[-1].set_xlim(xlims)
axs[-1].set_ylim(ylims)

axs[-1].set_xlabel('$x/D$')
for ax in axs:
    ax.set_ylabel('$y/D$')

plt.show()

#%% Subtract flowdatas
SWS = both.flowdata.interp(x = upstream.flowdata.x, y = upstream.flowdata.y, z = upstream.flowdata.z) - upstream.flowdata - downstream.flowdata

#%% Plot
fig, ax = plt.subplots(1,1)

V_zh = SWS.V.interp(z = 90.0)

ax.contourf(SWS.x/both.wf.D,
            SWS.y/both.wf.D,
            V_zh.T/both.wf.Uinf,
            cmap='jet')

ax.set_xlim(xlims)
ax.set_ylim(ylims)

plt.show()

#%% Analytical

both_a, _, _, U_h = both.analytical_solution('original', near_wake_correction=False)
both_a_nwc, _, _, _ = both.analytical_solution('original', near_wake_correction=True)

upstream_a, _, _, _ = upstream.analytical_solution('original', near_wake_correction=False)
upstream_a_nwc, _, _, _ = upstream.analytical_solution('original', near_wake_correction=True)

downstream_a, _, _, _ = downstream.analytical_solution('original', near_wake_correction=False)
downstream_a_nwc, _, _, _ = downstream.analytical_solution('original', near_wake_correction=True)

SWS_a = both_a.interp(x = upstream_a.x, y = upstream_a.y, z = upstream_a.z) - upstream_a - downstream_a

#%% Plot
fig, ax = plt.subplots(1,1)

V_zh = both_a.interp(x = upstream_a.x, y = upstream_a.y, z = upstream_a.z).V.interp(z = 90.0)

ax.contourf(upstream_a.x/both.wf.D,
            upstream_a.y/both.wf.D,
            V_zh.T/both.wf.Uinf,
            cmap='jet')

# ax.set_xlim(xlims)
# ax.set_ylim(ylims)

plt.show()