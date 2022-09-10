# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 16:04:19 2022

@author: nilsg
"""

#%% Module imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from post_utils import windTurbine, windFarm, flowcase, set_size, draw_AD
import copy

# Edit the font, font size, and axes width
plt.rcParams.update({
  "text.usetex": True,
  "font.family": "Tex Gyre Adventor"
})
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1

data_path = 'C:/Users/nilsg/OneDrive/Documents/EWEM/Thesis/NUMERICAL/PyWakeEllipSys/flowdata/'

fig_path = 'C:/Users/nilsg/Dropbox/Apps/Overleaf/Thesis/figures/06_modelPredictions/'

textwidth = 448.0 # [pts]

cmap_U = 'jet_r'
cmap_V = 'jet'

U_min = -0.05
U_max = 0.20
V_min = -0.025
V_max = 0.025
nlevels = 10

#%% Functions
def vel_zh(fd, zh):
    U_zh = fd.U.interp(z=zh)
    V_zh = fd.V.interp(z=zh)
    return U_zh, V_zh

#%% Wind turbine & wind farm objects

NREL5MW = windTurbine(D=126.0, zh=90.0, TSR=7.5)

wf_2WT_template = windFarm(Uinf=8,
              ti=0.06,
              x_wt=[0,7],
              y_wt=[0]*2,
              wts=[NREL5MW]*2,
              yaws=[25,0],
              CTs=[0.8]*2,
              wrs=[True,True])

wf_3WT_template = windFarm(Uinf=8,
              ti=0.06,
              x_wt=[0,7,14],
              y_wt=[0]*3,
              wts=[NREL5MW]*3,
              yaws=[25,25,0],
              CTs=[0.8]*3,
              wrs=[True]*3)

wf_4WT_template = windFarm(Uinf=8,
              ti=0.06,
              x_wt=[0,7,14,21],
              y_wt=[0]*4,
              wts=[NREL5MW]*4,
              yaws=[25,25,25,0],
              CTs=[0.8]*4,
              wrs=[True]*4)

wf_aligned = windFarm(Uinf=8,
                      ti=0.06,
                      x_wt=[ 0,  0,  0,
                             7,  7,  7,
                            14, 14, 14,
                            21, 21, 21,
                            28, 28, 28],
                      y_wt=[ 4,  -4,  0,
                             4,  -4,  0,
                             4,  -4,  0,
                             4,  -4,  0,
                             4,  -4,  0],
                      wts=[NREL5MW]*15,
                      yaws=[25]*15,
                      CTs=[0.8]*15,
                      wrs=[True]*15)

wf_slanted = windFarm(Uinf=8,
                      ti=0.06,
                      x_wt=[ 0,  0,  0,
                             7,  7,  7,
                            14, 14, 14,
                            21, 21, 21,
                            28, 28, 28],
                      y_wt=[ 4.00, -4.00,  0.00,
                             4.75, -3.25,  0.75,
                             5.50, -2.50,  1.50,
                             6.25, -1.75,  2.25,
                             7.00, -1.00,  3.00],
                      wts=[NREL5MW]*15,
                      yaws=[25]*15,
                      CTs=[0.8]*15,
                      wrs=[True]*15)

#%% Import flowcases
twoWT = flowcase(dir_path=data_path + '2WT/25_0/',
                 wf=copy.copy(wf_2WT_template),
                 cD=8)

threeWT = flowcase(dir_path=data_path + '3WT/25_25_0/',
                 wf=copy.copy(wf_3WT_template),
                 cD=8)

fourWT = flowcase(dir_path=data_path + '4WT/25_25_25_0/',
                 wf=copy.copy(wf_4WT_template),
                 cD=8)

WF_aligned = flowcase(dir_path=data_path + 'WF/aligned/on/',
                 wf=copy.copy(wf_aligned),
                 cD=8)

WF_slanted = flowcase(dir_path=data_path + 'WF/slanted/on/',
                 wf=copy.copy(wf_slanted),
                 cD=8)

flowcases = [twoWT, threeWT, fourWT, WF_aligned, WF_slanted]

#%% Flowcase savefig labels
fc_labels = ['2WT/25_0/', '3WT/25_25_0/', '4WT/25_25_25_0/', 'WF/aligned/on/', 'WF/slanted/on/']

P_total = np.zeros((3, len(flowcases)))

for i_fc, fc in enumerate([flowcases[0]]):
    #%% Shift turbine coordinates to match analytical results
    if fc.wf.n_wt < 10:
        fc.flowdata = fc.flowdata.assign_coords(x=(fc.flowdata.x + 3.5*(fc.wf.n_wt-1)*fc.wf.D))
    elif fc == WF_slanted:
        fc.flowdata = fc.flowdata.assign_coords(x=(fc.flowdata.x + 14*fc.wf.D), y=(fc.flowdata.y + 1.5*fc.wf.D)) # Translate coordinates
    else:
        fc.flowdata = fc.flowdata.assign_coords(x=(fc.flowdata.x + 3.5*(fc.wf.n_wt/3-1)*fc.wf.D))
        
    #%% Calculate wind farm power from PyWakeEllipSys results
    P, T = fc.powerAndThrust()
    P_total[0, i_fc] = sum(P)
    
    #%% Just streamwise analytical solution
    flowdata, flowdata_def, P, U_h = fc.streamwiseSolution(method = 'original')
    P_total[2, i_fc] = sum(P)
    
    #%% Hub-height velocities
    U_zh = flowdata.U.interp(z=fc.wf.zh)
    
    #%% Plotting - streamwise velocity
    # fraction of textwidth
    fraction = 1
    
    # Axes limits
    xlims = (fc.wf.x_wt[0]-2, fc.wf.x_wt[-1]+20)
    ylims = (min(fc.wf.y_wt)-3, max(fc.wf.y_wt)+3)
    
    # Contourf limits
    cmin = 0.55
    cmax = 1.05
    
    # Subplot labels
    subplot_labels = ['a) RANS', 'b) Analytical']
    
    # Create figure
    fig, axs = plt.subplots(2, 1, figsize=set_size(textwidth, fraction), sharex=True)
    
    # PyWakeEllipSys results
    p = axs[0].contourf(fc.flowdata.x/fc.wf.D,
                        fc.flowdata.y/fc.wf.D,
                        fc.U_zh.T/U_h,
                        cmap=cmap_V,
                        levels=np.linspace(cmin,cmax,nlevels+1),
                        vmin=cmin,
                        vmax=cmax,
                        extend='neither')
    axs[0].set_ylabel('$y/D$')
    axs[0].set_xlim(xlims)
    axs[0].set_ylim(ylims)
    
    # Analytical results
    p = axs[1].contourf(fc.flowdata.x/fc.wf.D,
                        fc.flowdata.y/fc.wf.D,
                        U_zh.T/U_h,
                        cmap=cmap_V,
                        levels=np.linspace(cmin,cmax,nlevels+1),
                        vmin=cmin,
                        vmax=cmax,
                        extend='neither')
    
    # Set axes limits and labels
    axs[1].set_xlabel('$x/D$')
    axs[1].set_ylabel('$y/D$')
    axs[1].set_xlim(xlims)
    axs[1].set_ylim(ylims)
    
    # Draw ADs
    for i in range(len(axs)):
        for i_t in range(fc.wf.n_wt):
            draw_AD(axs[i], 'top', fc.wf.x_wt[i_t], fc.wf.y_wt[i_t], fc.wf.D/fc.wf.D, fc.wf.yaws[i_t])
            
    # Add subplot labels
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    for i, ax in enumerate(axs):
        ax.text(0.06, 0.97, 
                     subplot_labels[i],
                     transform = ax.transAxes + trans,
                fontsize='medium', va='bottom', style='italic')
    
    # Add colourbar
    plt.subplots_adjust(right=0.8,
                        hspace=0.3)
    cax = fig.add_axes([0.825, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(p, cax=cax)
    cbar.set_label('$U/U_h$')

    #%% New analytical solution
    flowdata, flowdata_def, P, U_h = fc.analytical_solution(method = 'original', near_wake_correction=True, removes=[fc.wf.n_wt-1])
    P_total[1, i_fc] = sum(P)

    #%% Hub height velocities
    U_zh, V_zh = vel_zh(flowdata, fc.wf.zh)
    U_def_zh, V_def_zh = vel_zh(flowdata_def[fc.wf.n_wt-1], fc.wf.zh)
    V_sur_zh = -V_def_zh

    #%% Plotting - streamwise velocity
    # fraction of textwidth
    fraction = 1
    
    # Axes limits
    xlims = (fc.wf.x_wt[0]-2, fc.wf.x_wt[-1]+20)
    ylims = (min(fc.wf.y_wt)-3, max(fc.wf.y_wt)+3)
    
    # Contourf limits
    cmin = 0.55
    cmax = 1.05
    
    # Subplot labels
    subplot_labels = ['a) RANS', 'b) Analytical']
    
    # Create figure
    fig, axs = plt.subplots(2, 1, figsize=set_size(textwidth, fraction), sharex=True)
    
    # PyWakeEllipSys results
    p = axs[0].contourf(fc.flowdata.x/fc.wf.D,
                        fc.flowdata.y/fc.wf.D,
                        fc.U_zh.T/U_h,
                        cmap=cmap_V,
                        levels=np.linspace(cmin,cmax,nlevels+1),
                        vmin=cmin,
                        vmax=cmax,
                        extend='neither')
    axs[0].set_ylabel('$y/D$')
    axs[0].set_xlim(xlims)
    axs[0].set_ylim(ylims)
    
    # Analytical results
    p = axs[1].contourf(fc.flowdata.x/fc.wf.D,
                        fc.flowdata.y/fc.wf.D,
                        U_zh.T/U_h,
                        cmap=cmap_V,
                        levels=np.linspace(cmin,cmax,nlevels+1),
                        vmin=cmin,
                        vmax=cmax,
                        extend='neither')
    
    # Set axes limits and labels
    axs[1].set_xlabel('$x/D$')
    axs[1].set_ylabel('$y/D$')
    axs[1].set_xlim(xlims)
    axs[1].set_ylim(ylims)
    
    # Draw ADs
    for i in range(len(axs)):
        for i_t in range(fc.wf.n_wt):
            draw_AD(axs[i], 'top', fc.wf.x_wt[i_t], fc.wf.y_wt[i_t], fc.wf.D/fc.wf.D, fc.wf.yaws[i_t])
            
    # Add subplot labels
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    for i, ax in enumerate(axs):
        ax.text(0.06, 0.97, 
                     subplot_labels[i],
                     transform = ax.transAxes + trans,
                fontsize='medium', va='bottom', style='italic')
    
    # Add colourbar
    plt.subplots_adjust(right=0.8,
                        hspace=0.3)
    cax = fig.add_axes([0.825, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(p, cax=cax)
    cbar.set_label('$U/U_h$')
    
    # Save figure
    fig.savefig(fig_path + fc_labels[i_fc] + 'U_zh.pdf', bbox_inches='tight')
    plt.show()
    
    #%% Plotting - streamwise velocity deficit (turbine 1)
    # fraction of textwidth
    fraction = 1
    
    # Axes limits
    xlims = (fc.wf.x_wt[0]-2, fc.wf.x_wt[-1]+20)
    ylims = (min(fc.wf.y_wt)-3, max(fc.wf.y_wt)+3)
    
    # Contourf limits
    # cmin = -0.05
    # cmax = 0.2
    
    # Subplot labels
    subplot_labels = ['a) RANS', 'b) Analytical']
    
    # Create figure
    fig, axs = plt.subplots(2, 1, figsize=set_size(textwidth, fraction), sharex=True)
    
    # PyWakeEllipSys results
    p = axs[0].contourf(fc.flowdata.x/fc.wf.D,
                        fc.flowdata.y/fc.wf.D,
                        fc.Udef_zh[fc.wf.n_wt-1].T/U_h,
                        cmap=cmap_U,
                        levels=np.linspace(U_min,U_max,nlevels+1),
                        vmin=-U_max,
                        vmax=U_max,
                        extend='neither')
    axs[0].set_ylabel('$y/D$')
    axs[0].set_xlim(xlims)
    axs[0].set_ylim(ylims)
    
    # Analytical results
    p = axs[1].contourf(fc.flowdata.x/fc.wf.D,
                        fc.flowdata.y/fc.wf.D,
                        U_def_zh.T/U_h,
                        cmap=cmap_U,
                        levels=np.linspace(U_min,U_max,nlevels+1),
                        vmin=-U_max,
                        vmax=U_max,
                        extend='neither')
    
    # Set axes limits and labels
    axs[1].set_xlabel('$x/D$')
    axs[1].set_ylabel('$y/D$')
    axs[1].set_xlim(xlims)
    axs[1].set_ylim(ylims)
    
    # Draw ADs
    for i in range(len(axs)):
        for i_t in range(fc.wf.n_wt):
            draw_AD(axs[i], 'top', fc.wf.x_wt[i_t], fc.wf.y_wt[i_t], fc.wf.D/fc.wf.D, fc.wf.yaws[i_t])
            
    # Add subplot labels
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    for i, ax in enumerate(axs):
        ax.text(0.06, 0.97, 
                     subplot_labels[i],
                     transform = ax.transAxes + trans,
                fontsize='medium', va='bottom', style='italic')
    
    # Add colourbar
    plt.subplots_adjust(right=0.8,
                        hspace=0.3)
    cax = fig.add_axes([0.825, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(p, cax=cax)
    cbar.set_label('$(U_{n-1} - U_{n})/U_h$')
    
    # Save figure
    fig.savefig(fig_path + fc_labels[i_fc] + 'Udef_zh.pdf', bbox_inches='tight')
    plt.show()
    
    #%% Plotting - lateral velocity
    # fraction of textwidth
    fraction = 1
    
    # Axes limits
    xlims = (fc.wf.x_wt[0]-2, fc.wf.x_wt[-1]+20)
    ylims = (min(fc.wf.y_wt)-3, max(fc.wf.y_wt)+3)
    
    # Contourf limits
    cmin = -0.05
    cmax = 0.35
    
    # Subplot labels
    subplot_labels = ['a) RANS', 'b) Analytical']
    
    # Create figure
    fig, axs = plt.subplots(2, 1, figsize=set_size(textwidth, fraction), sharex=True)
    
    # PyWakeEllipSys results
    p = axs[0].contourf(fc.flowdata.x/fc.wf.D,
                        fc.flowdata.y/fc.wf.D,
                        fc.V_zh.T/U_h,
                        levels=np.linspace(cmin,cmax,nlevels+1),
                        vmin=cmin,
                        vmax=cmax,
                        cmap=cmap_V,
                        extend='neither')
    
    # Set axes limits and labels
    axs[0].set_ylabel('$y/D$')
    axs[0].set_xlim(xlims)
    axs[0].set_ylim(ylims)
    
    # Analytical results
    p = axs[1].contourf(fc.flowdata.x/fc.wf.D,
                        fc.flowdata.y/fc.wf.D,
                        V_zh.T/U_h,
                        levels=np.linspace(cmin,cmax,nlevels+1),
                        vmin=cmin,
                        vmax=cmax,
                        cmap=cmap_V,
                        extend='neither')
    
    # Set axes limits and labels
    axs[1].set_xlabel('$x/D$')
    axs[1].set_ylabel('$y/D$')
    axs[1].set_xlim(xlims)
    axs[1].set_ylim(ylims)
    
    # Draw ADs
    for i in range(len(axs)):
        for i_t in range(fc.wf.n_wt):
            draw_AD(axs[i], 'top', fc.wf.x_wt[i_t], fc.wf.y_wt[i_t], fc.wf.D/fc.wf.D, fc.wf.yaws[i_t])
    
    # Add subplot labels
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    for i, ax in enumerate(axs):
        ax.text(0.06, 0.97, 
                     subplot_labels[i],
                     transform = ax.transAxes + trans,
                fontsize='medium', va='bottom', style='italic')
    
    # Add colourbar
    plt.subplots_adjust(right=0.8,
                        hspace=0.3)
    cax = fig.add_axes([0.825, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(p, cax=cax)
    cbar.set_label('$V/U_h$')
    
    # Save figure
    fig.savefig(fig_path + fc_labels[i_fc] + 'V_zh.pdf', bbox_inches='tight')
    plt.show()
    
    #%% Plotting - lateral velocity surplus
    # fraction of textwidth
    fraction = 1
    
    # Axes limits
    xlims = (fc.wf.x_wt[0]-2, fc.wf.x_wt[-1]+8)
    ylims = (min(fc.wf.y_wt)-2, max(fc.wf.y_wt)+2)
    
    # Contourf limits
    cmin = -0.01
    cmax = 0.01
    
    # Subplot labels
    subplot_labels = ['a) RANS', 'b) Analytical']
    
    # Create figure
    fig, axs = plt.subplots(2, 1, figsize=set_size(textwidth, fraction), sharex=True)
    
    # PyWakeEllipSys results
    p = axs[0].contourf(fc.flowdata.x/fc.wf.D,
                        fc.flowdata.y/fc.wf.D,
                        fc.Vdef_zh[fc.wf.n_wt-1].T/U_h,
                        levels=np.linspace(cmin,cmax,nlevels+1),
                        vmin=cmin,
                        vmax=cmax,
                        cmap=cmap_V,
                        extend='neither')
    
    # Set axes limits and labels
    axs[0].set_ylabel('$y/D$')
    axs[0].set_xlim(xlims)
    axs[0].set_ylim(ylims)
    
    # Analytical results
    p = axs[1].contourf(fc.flowdata.x/fc.wf.D,
                        fc.flowdata.y/fc.wf.D,
                        V_sur_zh.T/U_h,
                        levels=np.linspace(cmin,cmax,nlevels+1),
                        vmin=cmin,
                        vmax=cmax,
                        cmap=cmap_V,
                        extend='neither')
    
    # Set axes limits and labels
    axs[1].set_xlabel('$x/D$')
    axs[1].set_ylabel('$y/D$')
    axs[1].set_xlim(xlims)
    axs[1].set_ylim(ylims)
    
    # Draw ADs
    for i in range(len(axs)):
        for i_t in range(fc.wf.n_wt):
            draw_AD(axs[i], 'top', fc.wf.x_wt[i_t], fc.wf.y_wt[i_t], fc.wf.D/fc.wf.D, fc.wf.yaws[i_t])
    
    # Add subplot labels
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    for i, ax in enumerate(axs):
        ax.text(0.06, 0.97, 
                     subplot_labels[i],
                     transform = ax.transAxes + trans,
                fontsize='medium', va='bottom', style='italic')
    
    # Add colourbar
    plt.subplots_adjust(right=0.8,
                        hspace=0.3)
    cax = fig.add_axes([0.825, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(p, cax=cax)
    cbar.set_label('$(V_{n} - V_{n-1})/U_h$')
    
    # Save figure
    fig.savefig(fig_path + fc_labels[i_fc] + 'Vdef_zh.pdf', bbox_inches='tight')
    plt.show()
    
#%% Plot wind farm powers
P_total = P_total/P_total[0,:]
print(P_total)