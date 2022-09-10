# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 14:07:37 2022

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

#%% Settings
V_min = -0.05
V_max = 0.2
nlevels = 10

U_def_min = -0.1
U_def_max = 0.5

np.linspace(U_def_min, U_def_max, nlevels+1)

n_x_D = 80

#%% Functions
def vel_zh(fd, zh):
    U_zh = fd.U.interp(z=zh)
    V_zh = fd.V.interp(z=zh)
    return U_zh, V_zh

def vel_pr_zh(flow_case, vel_zh, pos, WT):
    return vel_zh.interp(x = flow_case.wf.x_wt[WT]*flow_case.wf.D + pos*flow_case.wf.D)

def vel_yh(fd, yh):
    U_yh = fd.U.interp(y=yh)
    V_yh = fd.V.interp(y=yh)
    return U_yh, V_yh

def vel_pr_yh(flow_case, vel_yh, pos, WT):
    return vel_yh.interp(x = flow_case.wf.x_wt[WT]*flow_case.wf.D + pos*flow_case.wf.D)

#%% Wind turbine & wind farm objects

NREL5MW = windTurbine(D=126.0, zh=90.0, TSR=7.5)

wf_1WT_template = windFarm(Uinf=8,
              ti=0.06,
              x_wt=[0],
              y_wt=[0],
              wts=[NREL5MW],
              yaws=[25],
              CTs=[0.8],
              wrs=[True])

wf_1WT_ny_template = windFarm(Uinf=8,
              ti=0.06,
              x_wt=[0],
              y_wt=[0],
              wts=[NREL5MW],
              yaws=[0],
              CTs=[0.8],
              wrs=[True])

#%% Import flowcase
oneWT_ny = flowcase(dir_path = data_path + 'singleTurbine/0/',
                 wf = copy(wf_1WT_template),
                 cD = 8)

#%% Analytical solution
flowdata, _, _, U_h = oneWT_ny.streamwiseSolution(method='original')

#%% Hub height velocities
U_zh = flowdata.U.interp(z=oneWT_ny.wf.zh)

#%% Single Turbine: Figure 1
# Fraction of textwidth
fraction = 1

# Axes limits
xlims = (-2, 8)
ylims = (-2, 2)

# Downstream positions
x_Ds = [-1, 1, 3, 5, 7]

# Subplot labels
subplot_labels = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)']

# Other labels
labels = ['RANS', 'Bastankhah et al.']

# Create figure and axes object
fig = plt.figure(figsize=set_size(textwidth,
                                  fraction,
                                  height_adjust=1.33))

# Make outer gridspec
outer = gs.GridSpec(2, 1, figure=fig,
              height_ratios=[2,1], hspace = 0.35, right=0.8)

# Make nested gridspecs
gs1 = gs.GridSpecFromSubplotSpec(2, 1,
                                       subplot_spec = outer[0], hspace=0.3)
gs2 = gs.GridSpecFromSubplotSpec(1, len(x_Ds),
                                       subplot_spec = outer[1])

# Create axes objects
gs1_axs = []
for grs in gs1:
    gs1_axs.append(plt.subplot(grs))
gs2_axs = []
for grs in gs2:
    gs2_axs.append(plt.subplot(grs))

# First two share x and hide ticks
gs1_axs[0].sharex(gs1_axs[1])
plt.setp(gs1_axs[0].get_xticklabels(), visible=False)

# Share y on bottom row
for ax in gs2_axs[1:]:
    ax.sharex(gs2_axs[0])
    plt.setp(ax.get_yticklabels(), visible=False)

# Plot contours of lateral velocity at hub height
p = gs1_axs[0].contourf(oneWT_ny.U_zh.x/oneWT_ny.wf.D,
                   oneWT_ny.U_zh.y/oneWT_ny.wf.D,
                   1 - (oneWT_ny.U_zh.T/oneWT_ny.wf.Uinf),
                    levels=np.linspace(U_def_min,U_def_max,nlevels+1),
                    vmin=U_def_min,
                    vmax=U_def_max,
                   cmap='jet')
p = gs1_axs[1].contourf(oneWT_ny.flowdata.x/oneWT_ny.wf.D,
                   oneWT_ny.flowdata.y/oneWT_ny.wf.D,
                   1 - (U_zh.T/U_h),
                    levels=np.linspace(U_def_min,U_def_max,nlevels+1),
                    vmin=U_def_min,
                    vmax=U_def_max,
                   cmap='jet')

for i, ax in enumerate(gs1_axs):
    # Add actuator discs
    draw_AD(ax,
            view='top',
            x=oneWT_ny.wf.x_AD[0]/oneWT_ny.wf.D,
            y=oneWT_ny.wf.y_AD[0]/oneWT_ny.wf.D,
            D=oneWT_ny.wf.D/oneWT_ny.wf.D,
            yaw=0)
    
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax.text(0.07, 0.925, subplot_labels[i] + ' ' + labels[i], transform=ax.transAxes + trans,
            fontsize='medium', va='bottom',  style='italic')

    # Set axes limits
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    
    # Set axes labels
    ax.set_ylabel('$y/D$')
gs1_axs[1].set_xlabel('$x/D$', labelpad=-1.5)

# Add colourbar
cax  = fig.add_axes([0.825, 0.45, 0.03, 0.42])
cbar = fig.colorbar(p, cax=cax)
cbar.set_label('$1-(U/U_{\infty,h})$')

for ip, x_D in enumerate(x_Ds):
    # Add lines to contour plots to show planes
    for ax in gs1_axs:
        ax.axvline(x_D, color='k', ls='-.')
        
    # Extract profile velocities
    U_pr = oneWT_ny.velocityProfile('Udef', x_D, WT=0)
    U_pr_a = vel_pr_zh(oneWT_ny, U_h - U_zh, x_D, WT=0)
    
    # Plot profiles
    gs2_axs[ip].plot(U_pr/oneWT_ny.wf.Uinf, U_pr.y/oneWT_ny.wf.D, color='k', label='RANS')
    gs2_axs[ip].plot(U_pr_a/U_h, U_pr_a.y/oneWT_ny.wf.D, color='r', label='Analytical')
    
    # Add axes limits and labels
    gs2_axs[ip].set_ylim(ylims)
    gs2_axs[ip].set_xlabel('$1-\\frac{U}{U_{\infty,h}}$', fontsize=12)
    
    # Label subplots
    lab = r'$%sD$' % x_D
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    gs2_axs[ip].text(0.375, 0.925, subplot_labels[ip+2] + ' ' + lab, transform=gs2_axs[ip].transAxes + trans,
            fontsize='medium', va='bottom',  style='italic')

# Add y-label
gs2_axs[0].set_ylabel('$y/D$')

# Add legend
gs2_axs[-1].legend(loc='center left', bbox_to_anchor=(1.05, 0.5), frameon=False)

fig.savefig(fig_path + 'singleTurbine/U_zh_B' + '.pdf', bbox_inches='tight')

plt.show()

#%% Single Turbine: Figure 2
# Fraction of textwidth
fraction = 1

# Axes limits
xlims = (-2, 8)
zlims = (0, max(oneWT_ny.flowdata.z/oneWT_ny.wf.zh))

# Downstream positions
x_Ds = [-1, 1, 3, 5, 7]

# Subplot labels
subplot_labels = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)']

# Other labels
labels = ['RANS', 'Analytical']

# Create figure and axes object
fig = plt.figure(figsize=set_size(textwidth,
                                  fraction,
                                  height_adjust=1.33))

# Make outer gridspec
outer = gs.GridSpec(2, 1, figure=fig,
              height_ratios=[2,1], hspace = 0.35, right=0.8)

# Make nested gridspecs
gs1 = gs.GridSpecFromSubplotSpec(2, 1,
                                       subplot_spec = outer[0], hspace=0.3)
gs2 = gs.GridSpecFromSubplotSpec(1, len(x_Ds),
                                       subplot_spec = outer[1])

# Create axes objects
gs1_axs = []
for grs in gs1:
    gs1_axs.append(plt.subplot(grs))
gs2_axs = []
for grs in gs2:
    gs2_axs.append(plt.subplot(grs))

# First two share x and hide ticks
gs1_axs[0].sharex(gs1_axs[1])
plt.setp(gs1_axs[0].get_xticklabels(), visible=False)

# Share y on bottom row
for ax in gs2_axs[1:]:
    ax.sharex(gs2_axs[0])
    plt.setp(ax.get_yticklabels(), visible=False)
    
# Create x-z planes
oneWT_ny.U_yh = oneWT_ny.U.interp(y=0)
U_yh = flowdata.U.interp(y=0)

# Plot contours of lateral velocity at hub height
p = gs1_axs[0].contourf(oneWT_ny.U_yh.x/oneWT_ny.wf.D,
                   oneWT_ny.U_yh.z/oneWT_ny.wf.zh,
                   1-(oneWT_ny.U_yh.T/oneWT_ny.wf.Uinf),
                   levels=np.linspace(U_def_min,U_def_max,nlevels+1),
                   vmin=U_def_min,
                   vmax=U_def_max,
                   cmap='jet')
p = gs1_axs[1].contourf(oneWT_ny.flowdata.x/oneWT_ny.wf.D,
                   oneWT_ny.flowdata.z/oneWT_ny.wf.zh,
                   1-(U_yh.T/U_h),
                   levels=np.linspace(U_def_min,U_def_max,nlevels+1),
                   vmin=U_def_min,
                   vmax=U_def_max,
                   cmap='jet')

for i, ax in enumerate(gs1_axs):
    # Add actuator discs
    draw_AD(ax,
            view='side',
            x=oneWT_ny.wf.x_AD[0]/oneWT_ny.wf.D,
            y=oneWT_ny.wf.z_AD[0]/oneWT_ny.wf.zh,
            D=oneWT_ny.wf.D/oneWT_ny.wf.D,
            yaw=0)
    
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax.text(0.07, 0.925, subplot_labels[i] + ' ' + labels[i], transform=ax.transAxes + trans,
            fontsize='medium', va='bottom',  style='italic')

    # Set axes limits
    ax.set_xlim(xlims)
    ax.set_ylim(zlims)
    
    # Set axes labels
    ax.set_ylabel('$z/z_h$')
gs1_axs[1].set_xlabel('$x/D$', labelpad=-1.5)

# Add colourbar
cax  = fig.add_axes([0.825, 0.45, 0.03, 0.42])
cbar = fig.colorbar(p, cax=cax)
cbar.set_label('$1-(U/U_{\infty,h})$')

for ip, x_D in enumerate(x_Ds):
    # Add lines to contour plots to show planes
    for ax in gs1_axs:
        ax.axvline(x_D, color='k', ls='-.')
        
    # Extract profile velocities
    U_pr = vel_pr_yh(oneWT_ny,
                   oneWT_ny.U_yh,
                   x_D,
                   0)
    U_pr_a = vel_pr_yh(oneWT_ny,
                   U_yh,
                   x_D,
                   0)
    
    # Plot profiles
    gs2_axs[ip].plot(1-(U_pr/oneWT_ny.wf.Uinf),
                     U_pr.z/oneWT_ny.wf.zh,
                     color='k',
                     label='RANS')
    gs2_axs[ip].plot(1-(U_pr_a/U_h),
                     U_pr_a.z/oneWT_ny.wf.zh,
                     color='r',
                     label='Bastankhah et al.')
    
    # Add axes limits and labels
    gs2_axs[ip].set_ylim(zlims)
    gs2_axs[ip].set_xlabel('$1-\\frac{U}{U_{\infty,h}}$', fontsize=12)
    
    # Label subplots
    lab = r'$%sD$' % x_D
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    gs2_axs[ip].text(0.375, 0.925, subplot_labels[ip+2] + ' ' + lab, transform=gs2_axs[ip].transAxes + trans,
            fontsize='medium', va='bottom',  style='italic')
    
# Add y-label
gs2_axs[0].set_ylabel('$z/z_h$')

# Add legend
gs2_axs[-1].legend(loc='center left', bbox_to_anchor=(1.05, 0.5), frameon=False)

fig.savefig(fig_path + 'singleTurbine/U_yh_B' + '.pdf', bbox_inches='tight')

plt.show()

#%% Import flowcases
oneWT = flowcase(dir_path = data_path + 'singleTurbine/25/',
                 wf = copy(wf_1WT_template),
                 cD = 8)

#%% Analytical solution
flowdata, _, _, U_h = oneWT.analytical_solution(method='original', near_wake_correction=False)

#%% Hub height velocities
U_zh, V_zh = vel_zh(flowdata, oneWT.wf.zh)
U_yh, V_yh = vel_yh(flowdata, yh=0)

#%% Single Turbine: Figure 1
# Fraction of textwidth
fraction = 1

# Axes limits
xlims = (-2, 8)
ylims = (-2, 2)

# Downstream positions
x_Ds = [-1, 1, 3, 5, 7]

# Subplot labels
subplot_labels = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)']

# Other labels
labels = ['RANS', 'Analytical']

# Create figure and axes object
fig = plt.figure(figsize=set_size(textwidth,
                                  fraction,
                                  height_adjust=1.33))

# Make outer gridspec
outer = gs.GridSpec(2, 1, figure=fig,
              height_ratios=[2,1], hspace = 0.35, right=0.8)

# Make nested gridspecs
gs1 = gs.GridSpecFromSubplotSpec(2, 1,
                                       subplot_spec = outer[0], hspace=0.3)
gs2 = gs.GridSpecFromSubplotSpec(1, len(x_Ds),
                                       subplot_spec = outer[1])

# Create axes objects
gs1_axs = []
for grs in gs1:
    gs1_axs.append(plt.subplot(grs))
gs2_axs = []
for grs in gs2:
    gs2_axs.append(plt.subplot(grs))

# First two share x and hide ticks
gs1_axs[0].sharex(gs1_axs[1])
plt.setp(gs1_axs[0].get_xticklabels(), visible=False)

# Share y on bottom row
for ax in gs2_axs[1:]:
    ax.sharex(gs2_axs[0])
    plt.setp(ax.get_yticklabels(), visible=False)

# Plot contours of lateral velocity at hub height
p = gs1_axs[0].contourf(oneWT.V_zh.x/oneWT.wf.D,
                   oneWT.V_zh.y/oneWT.wf.D,
                   oneWT.V_zh.T/oneWT.wf.Uinf,
                   levels=np.linspace(V_min,V_max,nlevels+1),
                   vmin=V_min,
                   vmax=V_max,
                   cmap='jet')
p = gs1_axs[1].contourf(oneWT.flowdata.x/oneWT.wf.D,
                   oneWT.flowdata.y/oneWT.wf.D,
                   V_zh.T/U_h,
                   levels=np.linspace(V_min,V_max,nlevels+1),
                   vmin=V_min,
                   vmax=V_max,
                   cmap='jet')

for i, ax in enumerate(gs1_axs):
    # Add actuator discs
    draw_AD(ax,
            view='top',
            x=oneWT.wf.x_AD[0]/oneWT.wf.D,
            y=oneWT.wf.y_AD[0]/oneWT.wf.D,
            D=oneWT.wf.D/oneWT.wf.D,
            yaw=oneWT.wf.yaws[0])
    
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax.text(0.07, 0.925, subplot_labels[i] + ' ' + labels[i], transform=ax.transAxes + trans,
            fontsize='medium', va='bottom',  style='italic')

    # Set axes limits
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    
    # Set axes labels
    ax.set_ylabel('$y/D$')
gs1_axs[1].set_xlabel('$x/D$', labelpad=-1.5)

# Add colourbar
cax  = fig.add_axes([0.825, 0.45, 0.03, 0.42])
cbar = fig.colorbar(p, cax=cax)
cbar.set_label('$V/U_{\infty,h}$')

for ip, x_D in enumerate(x_Ds):
    # Add lines to contour plots to show planes
    for ax in gs1_axs:
        ax.axvline(x_D, color='k', ls='-.')
        
    # Extract profile velocities
    V_pr = oneWT.velocityProfile('V', x_D, WT=0)
    V_pr_a = vel_pr_zh(oneWT, V_zh, x_D, WT=0)
    
    # Plot profiles
    gs2_axs[ip].plot(V_pr/oneWT.wf.Uinf, V_pr.y/oneWT.wf.D, color='k', label='RANS')
    gs2_axs[ip].plot(V_pr_a/U_h, V_pr_a.y/oneWT.wf.D, color='r', label='Analytical')
    
    # Add axes limits and labels
    gs2_axs[ip].set_ylim(ylims)
    gs2_axs[ip].set_xlabel('$V/U_{\infty,h}$')
    
    # Label subplots
    lab = r'$%sD$' % x_D
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    gs2_axs[ip].text(0.375, 0.925, subplot_labels[ip+2] + ' ' + lab, transform=gs2_axs[ip].transAxes + trans,
            fontsize='medium', va='bottom',  style='italic')

# Add y-label
gs2_axs[0].set_ylabel('$y/D$')

# Add legend
gs2_axs[-1].legend(loc='center left', bbox_to_anchor=(1.05, 0.5), frameon=False)

fig.savefig(fig_path + 'singleTurbine/V_zh' + '.pdf', bbox_inches='tight')

plt.show()

#%% Single Turbine: Figure 2
# Fraction of textwidth
fraction = 1

# Axes limits
xlims = (-2, 8)
zlims = (0, max(oneWT.flowdata.z/oneWT.wf.zh))

# Downstream positions
x_Ds = [-1, 1, 3, 5, 7]

# Subplot labels
subplot_labels = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)']

# Other labels
labels = ['RANS', 'Analytical']

# Create figure and axes object
fig = plt.figure(figsize=set_size(textwidth,
                                  fraction,
                                  height_adjust=1.33))

# Make outer gridspec
outer = gs.GridSpec(2, 1, figure=fig,
              height_ratios=[2,1], hspace = 0.35, right=0.8)

# Make nested gridspecs
gs1 = gs.GridSpecFromSubplotSpec(2, 1,
                                       subplot_spec = outer[0], hspace=0.3)
gs2 = gs.GridSpecFromSubplotSpec(1, len(x_Ds),
                                       subplot_spec = outer[1])

# Create axes objects
gs1_axs = []
for grs in gs1:
    gs1_axs.append(plt.subplot(grs))
gs2_axs = []
for grs in gs2:
    gs2_axs.append(plt.subplot(grs))

# First two share x and hide ticks
gs1_axs[0].sharex(gs1_axs[1])
plt.setp(gs1_axs[0].get_xticklabels(), visible=False)

# Share y on bottom row
for ax in gs2_axs[1:]:
    ax.sharex(gs2_axs[0])
    plt.setp(ax.get_yticklabels(), visible=False)
    
# Create x-z planes
_, oneWT.V_yh = vel_yh(oneWT.flowdata, yh=0)

# Plot contours of lateral velocity at hub height
p = gs1_axs[0].contourf(oneWT.V_yh.x/oneWT.wf.D,
                   oneWT.V_yh.z/oneWT.wf.zh,
                   oneWT.V_yh.T/oneWT.wf.Uinf,
                   levels=np.linspace(V_min,V_max,nlevels+1),
                   vmin=V_min,
                   vmax=V_max,
                   cmap='jet')
p = gs1_axs[1].contourf(oneWT.flowdata.x/oneWT.wf.D,
                   oneWT.flowdata.z/oneWT.wf.zh,
                   V_yh.T/U_h,
                   levels=np.linspace(V_min,V_max,nlevels+1),
                   vmin=V_min,
                   vmax=V_max,
                   cmap='jet')

for i, ax in enumerate(gs1_axs):
    # Add actuator discs
    draw_AD(ax,
            view='side',
            x=oneWT.wf.x_AD[0]/oneWT.wf.D,
            y=oneWT.wf.z_AD[0]/oneWT.wf.zh,
            D=oneWT.wf.D/oneWT.wf.D,
            yaw=oneWT.wf.yaws[0])
    
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax.text(0.07, 0.925, subplot_labels[i] + ' ' + labels[i], transform=ax.transAxes + trans,
            fontsize='medium', va='bottom',  style='italic')

    # Set axes limits
    ax.set_xlim(xlims)
    ax.set_ylim(zlims)
    
    # Set axes labels
    ax.set_ylabel('$z/z_h$')
gs1_axs[1].set_xlabel('$x/D$', labelpad=-1.5)

# Add colourbar
cax  = fig.add_axes([0.825, 0.45, 0.03, 0.42])
cbar = fig.colorbar(p, cax=cax)
cbar.set_label('$V/U_{\infty,h}$')

for ip, x_D in enumerate(x_Ds):
    # Add lines to contour plots to show planes
    for ax in gs1_axs:
        ax.axvline(x_D, color='k', ls='-.')
        
    # Extract profile velocities
    V_pr = vel_pr_yh(oneWT,
                   oneWT.V_yh,
                   x_D,
                   0)
    V_pr_a = vel_pr_yh(oneWT,
                   V_yh,
                   x_D,
                   0)
    
    # Plot profiles
    gs2_axs[ip].plot(V_pr/oneWT.wf.Uinf,
                     V_pr.z/oneWT.wf.zh,
                     color='k',
                     label='RANS')
    gs2_axs[ip].plot(V_pr_a/U_h,
                     V_pr_a.z/oneWT.wf.zh,
                     color='r',
                     label='Analytical')
    
    # Add axes limits and labels
    gs2_axs[ip].set_ylim(zlims)
    gs2_axs[ip].set_xlabel('$V/U_{\infty,h}$')
    
    # Label subplots
    lab = r'$%sD$' % x_D
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    gs2_axs[ip].text(0.375, 0.925, subplot_labels[ip+2] + ' ' + lab, transform=gs2_axs[ip].transAxes + trans,
            fontsize='medium', va='bottom',  style='italic')
    
# Add y-label
gs2_axs[0].set_ylabel('$z/z_h$')

# Add legend
gs2_axs[-1].legend(loc='center left', bbox_to_anchor=(1.05, 0.5), frameon=False)

fig.savefig(fig_path + 'singleTurbine/V_yh' + '.pdf', bbox_inches='tight')

plt.show()

#%% Near wake correction
flowdata_nwc, _, _, _ = oneWT.analytical_solution(method='original', near_wake_correction=True)

#%% Hub height velocities
U_zh_nwc, _ = vel_zh(flowdata_nwc, oneWT.wf.zh)

#%% Single Turbine: Figure 3
# Fraction of textwidth
fraction = 1

# Axes limits
xlims = (-2, 8)
ylims = (-3, 3)

# Wake centre method
wcm='Gaussian'

# Downstream positions
x_Ds = [-1, 1, 3, 5, 7]
x_Ds_2 = np.linspace(0, 8, n_x_D+1)

# Subplot labels
subplot_labels = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)']

# Other labels
labels = ['RANS', 'Analytical', 'Corrected']

# Create figure and axes object
fig = plt.figure(figsize=set_size(textwidth,
                                  fraction,
                                  height_adjust=1.5))

# Make outer gridspec
outer = gs.GridSpec(2, 1, figure=fig,
              height_ratios=[3,1], hspace = 0.275, right=0.8)

# Make nested gridspecs
gs1 = gs.GridSpecFromSubplotSpec(3, 1,
                                       subplot_spec = outer[0], hspace=0.325)
gs2 = gs.GridSpecFromSubplotSpec(1, 1,
                                       subplot_spec = outer[1])

# Create axes objects
gs1_axs = []
for grs in gs1:
    gs1_axs.append(plt.subplot(grs))
gs2_axs = []
for grs in gs2:
    gs2_axs.append(plt.subplot(grs))

# First three share x and hide ticks
gs1_axs[0].sharex(gs1_axs[2])
gs1_axs[1].sharex(gs1_axs[2])
plt.setp(gs1_axs[0].get_xticklabels(), visible=False)
plt.setp(gs1_axs[1].get_xticklabels(), visible=False)

# Plot contours of lateral velocity at hub height
p = gs1_axs[0].contourf(oneWT.U_zh.x/oneWT.wf.D,
                   oneWT.U_zh.y/oneWT.wf.D,
                   1 - (oneWT.U_zh.T/oneWT.wf.Uinf),
                   levels=np.linspace(U_def_min,U_def_max,nlevels+1),
                   vmin=U_def_min,
                   vmax=U_def_max,
                   cmap='jet')
p = gs1_axs[1].contourf(oneWT.flowdata.x/oneWT.wf.D,
                   oneWT.flowdata.y/oneWT.wf.D,
                   1 - (U_zh.T/U_h),
                   levels=np.linspace(U_def_min,U_def_max,nlevels+1),
                   vmin=U_def_min,
                   vmax=U_def_max,
                   cmap='jet')

p = gs1_axs[2].contourf(oneWT.flowdata.x/oneWT.wf.D,
                   oneWT.flowdata.y/oneWT.wf.D,
                   1 - (U_zh_nwc.T/U_h),
                   levels=np.linspace(U_def_min,U_def_max,nlevels+1),
                   vmin=U_def_min,
                   vmax=U_def_max,
                   cmap='jet')

for i, ax in enumerate(gs1_axs):
    # Add actuator discs
    draw_AD(ax,
            view='top',
            x=oneWT.wf.x_AD[0]/oneWT.wf.D,
            y=oneWT.wf.y_AD[0]/oneWT.wf.D,
            D=oneWT.wf.D/oneWT.wf.D,
            yaw=oneWT.wf.yaws[0])
    
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax.text(0.07, 0.925, subplot_labels[i] + ' ' + labels[i], transform=ax.transAxes + trans,
            fontsize='medium', va='bottom',  style='italic')

    # Set axes limits
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    
    # Set axes labels
    ax.set_ylabel('$y/D$')
gs1_axs[2].set_xlabel('$x/D$', labelpad=-1.5)

# Add colourbar
cax  = fig.add_axes([0.825, 0.38, 0.03, 0.5])
cbar = fig.colorbar(p, cax=cax)
cbar.set_label('$1- (U/U_{\infty,h})$')

centres = np.empty((3, len(x_Ds_2)))
for ip, x_D in enumerate(x_Ds_2):
    # Extract profile
    U_pr = oneWT.velocityProfile('Udef', x_D, WT=0)
    U_pr_a = U_h - vel_pr_zh(oneWT, U_zh, x_D, WT=0)
    U_pr_c = U_h - vel_pr_zh(oneWT, U_zh_nwc, x_D, WT=0)
    
    # Calculate wake centres
    centres[0,ip], _ = wakeCentre(wcm,
                              U_pr.y,
                              U_pr)
    centres[1,ip], _ = wakeCentre('max',
                              U_pr_a.y,
                              U_pr_a)
    centres[2,ip], _ = wakeCentre(wcm,
                              U_pr_c.y,
                              U_pr_c)
    
# Plot
gs2_axs[0].plot(x_Ds_2, centres[0,:]/oneWT.wf.D,
        color='k', label='RANS')
gs2_axs[0].plot(x_Ds_2, centres[1,:]/oneWT.wf.D,
        color='r', label='Analytical')
gs2_axs[0].plot(x_Ds_2, centres[2,:]/oneWT.wf.D,
        color='g', label='Corrected')
    
# Label subplots
trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
gs2_axs[0].text(0.07, 0.925, subplot_labels[3], transform=gs2_axs[0].transAxes + trans,
        fontsize='medium', va='bottom', style='italic')

# Axes limits
gs2_axs[0].set_xlim(xlims)

# Add axes labels
gs2_axs[0].set_xlabel('$x/D$', labelpad=-2)
gs2_axs[0].set_ylabel('$(y^{*,1}-y)/D$')

# Add legend
gs2_axs[0].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)

fig.savefig(fig_path + 'singleTurbine/wakeCentres' + '.pdf', bbox_inches='tight')

plt.show()

#%% Debugging
# for ip, x_D in enumerate(x_Ds_2):
#     U_pr_a = U_h - vel_pr_zh(oneWT, U_zh, x_D, WT=0)
#     # centres[1,ip], _ = wakeCentre('max',
#                               # y_to_ys(U_pr_a.y, U_pr_a, 'Gaussian'),
#                               # U_pr_a)
    
#     plt.plot(U_pr_a.y/oneWT.wf.D,
#              U_pr_a, label=str(x_D))
#     plt.xlim((-3,3))
#     plt.legend()
    
#     plt.show()

# plt.plot(x_Ds_2, centres[1,:]/oneWT.wf.D)
# plt.show()

#%% Two-turbine
wf_2WT_25_0_template = windFarm(Uinf=8,
              ti=0.06,
              x_wt=[0,7],
              y_wt=[0]*2,
              wts=[NREL5MW]*2,
              yaws=[25,0],
              CTs=[0.8]*2,
              wrs=[True]*2)

#%% Import flowcase
twoWT_25_0 = flowcase(dir_path = data_path + '2WT/25_0/',
                 wf = copy(wf_2WT_25_0_template),
                 cD = 8)

fc = twoWT_25_0

twoWT_25_0.flowdata = twoWT_25_0.flowdata.assign_coords(x=(twoWT_25_0.flowdata.x + 3.5*(twoWT_25_0.wf.n_wt-1)*twoWT_25_0.wf.D))

#%% Analytical solution
flowdata_s, _, _, _ = twoWT_25_0.streamwiseSolution(method='original')

flowdata, flowdata_def, _, U_h = twoWT_25_0.analytical_solution(method='original', near_wake_correction=False, removes=[1])

flowdata_nwc, flowdata_def_nwc, _, _ = twoWT_25_0.analytical_solution(method='original', near_wake_correction=True, removes=[1])

#%% Hub height velocities
U_zh_s = flowdata_s.U.interp(z=twoWT_25_0.wf.zh)

U_zh, V_zh = vel_zh(flowdata, twoWT_25_0.wf.zh)
U_def_zh, V_def_zh = vel_zh(flowdata_def[twoWT_25_0.wf.n_wt-1], twoWT_25_0.wf.zh)
V_sur_zh = -V_def_zh

U_zh_nwc, V_zh_nwc = vel_zh(flowdata_nwc, twoWT_25_0.wf.zh)
U_def_zh_nwc, V_def_zh_nwc = vel_zh(flowdata_def_nwc[twoWT_25_0.wf.n_wt-1], twoWT_25_0.wf.zh)
V_sur_zh_nwc = -V_def_zh_nwc

#%% U_zh
# Fraction of textwidth
fraction = 1

# Axes limits
xlims = (-2, 7+8)
ylims = (-2, 2)

# Downstream positions
x_Ds = 7 + np.asarray([-1, 1, 3, 5, 7])

# Subplot labels
subplot_labels = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)']

# Other labels
labels = ['RANS', 'Bastankhah et al.', 'Analytical', 'Corrected']

# Create figure and axes object
fig = plt.figure(figsize=set_size(textwidth,
                                  fraction,
                                  height_adjust=1.75))

# Make outer gridspec
outer = gs.GridSpec(2, 1, figure=fig,
              height_ratios=[3.5,1], hspace = 0.275, right=0.8)

# Make nested gridspecs
gs1 = gs.GridSpecFromSubplotSpec(4, 1,
                                       subplot_spec = outer[0], hspace=0.375)
gs2 = gs.GridSpecFromSubplotSpec(1, len(x_Ds),
                                       subplot_spec = outer[1])

# Create axes objects
gs1_axs = []
for grs in gs1:
    gs1_axs.append(plt.subplot(grs))
gs2_axs = []
for grs in gs2:
    gs2_axs.append(plt.subplot(grs))

# First two share x and hide ticks
gs1_axs[0].sharex(gs1_axs[3])
gs1_axs[1].sharex(gs1_axs[3])
gs1_axs[2].sharex(gs1_axs[3])
plt.setp(gs1_axs[0].get_xticklabels(), visible=False)
plt.setp(gs1_axs[1].get_xticklabels(), visible=False)
plt.setp(gs1_axs[2].get_xticklabels(), visible=False)

# Share y on bottom row
for ax in gs2_axs[1:]:
    ax.sharex(gs2_axs[0])
    plt.setp(ax.get_yticklabels(), visible=False)

# Plot contours of lateral velocity at hub height
p = gs1_axs[0].contourf(twoWT_25_0.flowdata.x/twoWT_25_0.wf.D,
                   twoWT_25_0.flowdata.y/twoWT_25_0.wf.D,
                   1-twoWT_25_0.U_zh.T/twoWT_25_0.wf.Uinf,
                   levels=np.linspace(U_def_min,U_def_max,nlevels+1),
                   vmin=U_def_min,
                   vmax=U_def_max,
                   cmap='jet')
p = gs1_axs[1].contourf(twoWT_25_0.flowdata.x/twoWT_25_0.wf.D,
                   twoWT_25_0.flowdata.y/twoWT_25_0.wf.D,
                   1-U_zh_s.T/U_h,
                   levels=np.linspace(U_def_min,U_def_max,nlevels+1),
                   vmin=U_def_min,
                   vmax=U_def_max,
                   cmap='jet')
p = gs1_axs[2].contourf(twoWT_25_0.flowdata.x/twoWT_25_0.wf.D,
                   twoWT_25_0.flowdata.y/twoWT_25_0.wf.D,
                   1-U_zh.T/U_h,
                   levels=np.linspace(U_def_min,U_def_max,nlevels+1),
                   vmin=U_def_min,
                   vmax=U_def_max,
                   cmap='jet')
p = gs1_axs[3].contourf(twoWT_25_0.flowdata.x/twoWT_25_0.wf.D,
                   twoWT_25_0.flowdata.y/twoWT_25_0.wf.D,
                   1-U_zh_nwc.T/U_h,
                   levels=np.linspace(U_def_min,U_def_max,nlevels+1),
                   vmin=U_def_min,
                   vmax=U_def_max,
                   cmap='jet')

for i, ax in enumerate(gs1_axs):
    for i_t in range(twoWT_25_0.wf.n_wt):
        # Add actuator discs
        draw_AD(ax,
                view='top',
                x=twoWT_25_0.wf.x_wt[i_t],
                y=twoWT_25_0.wf.y_wt[i_t],
                D=twoWT_25_0.wf.D/twoWT_25_0.wf.D,
                yaw=twoWT_25_0.wf.yaws[i_t])
    
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax.text(0.07, 0.925, subplot_labels[i] + ' ' + labels[i], transform=ax.transAxes + trans,
            fontsize='medium', va='bottom',  style='italic')

    # Set axes limits
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    
    # Set axes labels
    ax.set_ylabel('$y/D$')
gs1_axs[3].set_xlabel('$x/D$', labelpad=-1.5)

# Add colourbar
cax  = fig.add_axes([0.825, 0.38, 0.03, 0.47])
cbar = fig.colorbar(p, cax=cax)
cbar.set_label('$1-(U/U_{\infty,h})$')

for ip, x_D in enumerate(x_Ds):
    # Add lines to contour plots to show planes
    for ax in gs1_axs:
        ax.axvline(x_D, color='k', ls='-.')
        
    # Extract profile velocities
    U_pr = twoWT_25_0.velocityProfile('Udef', x_D, WT=1)
    U_pr_s = vel_pr_zh(twoWT_25_0, U_zh_s, x_D, WT=0)
    U_pr_a = vel_pr_zh(twoWT_25_0, U_zh, x_D, WT=0)
    U_pr_nwc = vel_pr_zh(twoWT_25_0, U_zh_nwc, x_D, WT=0)
    
    # Plot profiles
    gs2_axs[ip].plot(U_pr/twoWT_25_0.wf.Uinf, twoWT_25_0.flowdata.y/twoWT_25_0.wf.D, color='k', label='RANS')
    gs2_axs[ip].plot(1-U_pr_s/twoWT_25_0.wf.Uinf, twoWT_25_0.flowdata.y/twoWT_25_0.wf.D, color='b', label='Bastankhah et al.')
    gs2_axs[ip].plot(1-U_pr_a/U_h, twoWT_25_0.flowdata.y/twoWT_25_0.wf.D, color='r', label='Analytical')
    gs2_axs[ip].plot(1-U_pr_nwc/U_h, twoWT_25_0.flowdata.y/twoWT_25_0.wf.D, color='g', label='Corrected')
    
    # Add axes limits and labels
    gs2_axs[ip].set_ylim(ylims)
    gs2_axs[ip].set_xlabel('$1-\\frac{U}{U_{\infty,h}}$', fontsize=12)
    
    # Label subplots
    lab = r'$%sD$' % x_D
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    gs2_axs[ip].text(0.375, 0.925, subplot_labels[ip+3] + ' ' + lab, transform=gs2_axs[ip].transAxes + trans,
            fontsize='medium', va='bottom',  style='italic')

# Add y-label
gs2_axs[0].set_ylabel('$y/D$')

# Add legend
gs2_axs[-1].legend(loc='center left', bbox_to_anchor=(1.05, 0.5), frameon=False)

fig.savefig(fig_path + 'twoTurbine/U_zh' + '.pdf', bbox_inches='tight')

plt.show()


#%% V_zh
# Fraction of textwidth
fraction = 1

# Axes limits
xlims = (-2, 7+8)
ylims = (-2, 2)

# Downstream positions
x_Ds = 7 + np.asarray([-1, 1, 3, 5, 7])

# Subplot labels
subplot_labels = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)']

# Other labels
labels = ['RANS', 'Analytical', 'Corrected']

# Create figure and axes object
fig = plt.figure(figsize=set_size(textwidth,
                                  fraction,
                                  height_adjust=1.33))

# Make outer gridspec
outer = gs.GridSpec(2, 1, figure=fig,
              height_ratios=[3,1], hspace = 0.35, right=0.8)

# Make nested gridspecs
gs1 = gs.GridSpecFromSubplotSpec(3, 1,
                                       subplot_spec = outer[0], hspace=0.4)
gs2 = gs.GridSpecFromSubplotSpec(1, len(x_Ds),
                                       subplot_spec = outer[1])

# Create axes objects
gs1_axs = []
for grs in gs1:
    gs1_axs.append(plt.subplot(grs))
gs2_axs = []
for grs in gs2:
    gs2_axs.append(plt.subplot(grs))

# First two share x and hide ticks
gs1_axs[0].sharex(gs1_axs[2])
gs1_axs[1].sharex(gs1_axs[2])
plt.setp(gs1_axs[0].get_xticklabels(), visible=False)
plt.setp(gs1_axs[1].get_xticklabels(), visible=False)

# Share y on bottom row
for ax in gs2_axs[1:]:
    ax.sharex(gs2_axs[0])
    plt.setp(ax.get_yticklabels(), visible=False)

# Plot contours of lateral velocity at hub height
p = gs1_axs[0].contourf(twoWT_25_0.flowdata.x/twoWT_25_0.wf.D,
                   twoWT_25_0.flowdata.y/twoWT_25_0.wf.D,
                   twoWT_25_0.V_zh.T/twoWT_25_0.wf.Uinf,
                   levels=np.linspace(V_min,V_max,nlevels+1),
                   vmin=V_min,
                   vmax=V_max,
                   cmap='jet')
p = gs1_axs[1].contourf(twoWT_25_0.flowdata.x/twoWT_25_0.wf.D,
                   twoWT_25_0.flowdata.y/twoWT_25_0.wf.D,
                   V_zh.T/U_h,
                   levels=np.linspace(V_min,V_max,nlevels+1),
                   vmin=V_min,
                   vmax=V_max,
                   cmap='jet')
p = gs1_axs[2].contourf(twoWT_25_0.flowdata.x/twoWT_25_0.wf.D,
                   twoWT_25_0.flowdata.y/twoWT_25_0.wf.D,
                   V_zh_nwc.T/U_h,
                   levels=np.linspace(V_min,V_max,nlevels+1),
                   vmin=V_min,
                   vmax=V_max,
                   cmap='jet')

for i, ax in enumerate(gs1_axs):
    for i_t in range(twoWT_25_0.wf.n_wt):
        # Add actuator discs
        draw_AD(ax,
                view='top',
                x=twoWT_25_0.wf.x_wt[i_t],
                y=twoWT_25_0.wf.y_wt[i_t],
                D=twoWT_25_0.wf.D/twoWT_25_0.wf.D,
                yaw=twoWT_25_0.wf.yaws[i_t])
    
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax.text(0.07, 0.9, subplot_labels[i] + ' ' + labels[i], transform=ax.transAxes + trans,
            fontsize='medium', va='bottom',  style='italic')

    # Set axes limits
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    
    # Set axes labels
    ax.set_ylabel('$y/D$')
gs1_axs[2].set_xlabel('$x/D$', labelpad=-1.5)

# Add colourbar
cax  = fig.add_axes([0.825, 0.4, 0.03, 0.47])
cbar = fig.colorbar(p, cax=cax)
cbar.set_label('$V/U_{\infty,h}$')

for ip, x_D in enumerate(x_Ds):
    # Add lines to contour plots to show planes
    for ax in gs1_axs:
        ax.axvline(x_D, color='k', ls='-.')
        
    # Extract profile velocities
    V_pr = twoWT_25_0.velocityProfile('V', x_D, WT=1)
    V_pr_a = vel_pr_zh(twoWT_25_0, V_zh, x_D, WT=0)
    V_pr_nwc = vel_pr_zh(twoWT_25_0, V_zh_nwc, x_D, WT=0)
    
    # Plot profiles
    gs2_axs[ip].plot(V_pr/twoWT_25_0.wf.Uinf, twoWT_25_0.flowdata.y/twoWT_25_0.wf.D, color='k', label='RANS')
    gs2_axs[ip].plot(V_pr_a/U_h, twoWT_25_0.flowdata.y/twoWT_25_0.wf.D, color='r', label='Analytical')
    gs2_axs[ip].plot(V_pr_nwc/U_h, twoWT_25_0.flowdata.y/twoWT_25_0.wf.D, color='g', label='Corrected')
    
    # Add axes limits and labels
    gs2_axs[ip].set_ylim(ylims)
    gs2_axs[ip].set_xlabel('$V/U_{\infty,h}$')
    
    # Label subplots
    lab = r'$%sD$' % x_D
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    gs2_axs[ip].text(0.375, 0.925, subplot_labels[ip+2] + ' ' + lab, transform=gs2_axs[ip].transAxes + trans,
            fontsize='medium', va='bottom',  style='italic')

# Add y-label
gs2_axs[0].set_ylabel('$y/D$')

# Add legend
gs2_axs[-1].legend(loc='center left', bbox_to_anchor=(1.05, 0.5), frameon=False)

fig.savefig(fig_path + 'twoTurbine/V_zh' + '.pdf', bbox_inches='tight')

plt.show()

#%% Plotting - lateral velocity surplus
# fraction of textwidth
fraction = 1

# Axes limits
xlims = (twoWT_25_0.wf.x_wt[0]-2, twoWT_25_0.wf.x_wt[-1]+8)
ylims = (min(twoWT_25_0.wf.y_wt)-2, max(twoWT_25_0.wf.y_wt)+2)

# Contourf limits
cmin = -0.01
cmax = 0.01

# Downstream positions
x_Ds = 7 + np.asarray([-1, 1, 3, 5, 7])

# Subplot labels
subplot_labels = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)']

# Other labels
labels = ['RANS', 'Analytical', 'Corrected']

# Create figure and axes object
fig = plt.figure(figsize=set_size(textwidth,
                                  fraction,
                                  height_adjust=1.33))

# Make outer gridspec
outer = gs.GridSpec(2, 1, figure=fig,
              height_ratios=[3,1], hspace = 0.35, right=0.8)

# Make nested gridspecs
gs1 = gs.GridSpecFromSubplotSpec(3, 1,
                                       subplot_spec = outer[0], hspace=0.4)
gs2 = gs.GridSpecFromSubplotSpec(1, len(x_Ds),
                                       subplot_spec = outer[1])

# Create axes objects
gs1_axs = []
for grs in gs1:
    gs1_axs.append(plt.subplot(grs))
gs2_axs = []
for grs in gs2:
    gs2_axs.append(plt.subplot(grs))

# First two share x and hide ticks
gs1_axs[0].sharex(gs1_axs[2])
gs1_axs[1].sharex(gs1_axs[2])
plt.setp(gs1_axs[0].get_xticklabels(), visible=False)
plt.setp(gs1_axs[1].get_xticklabels(), visible=False)

# Share y on bottom row
for ax in gs2_axs[1:]:
    ax.sharex(gs2_axs[0])
    plt.setp(ax.get_yticklabels(), visible=False)

# PyWakeEllipSys results
p = gs1_axs[0].contourf(twoWT_25_0.flowdata.x/twoWT_25_0.wf.D,
                    twoWT_25_0.flowdata.y/twoWT_25_0.wf.D,
                    twoWT_25_0.Vdef_zh[twoWT_25_0.wf.n_wt-1].T/U_h,
                    levels=np.linspace(cmin,cmax,nlevels+1),
                    vmin=cmin,
                    vmax=cmax,
                    cmap='jet',
                    extend='neither')

# Analytical results
p = gs1_axs[1].contourf(twoWT_25_0.flowdata.x/twoWT_25_0.wf.D,
                    twoWT_25_0.flowdata.y/twoWT_25_0.wf.D,
                    V_sur_zh.T/U_h,
                    levels=np.linspace(cmin,cmax,nlevels+1),
                    vmin=cmin,
                    vmax=cmax,
                    cmap='jet',
                    extend='neither')

p = gs1_axs[2].contourf(twoWT_25_0.flowdata.x/twoWT_25_0.wf.D,
                    twoWT_25_0.flowdata.y/twoWT_25_0.wf.D,
                    V_sur_zh_nwc.T/U_h,
                    levels=np.linspace(cmin,cmax,nlevels+1),
                    vmin=cmin,
                    vmax=cmax,
                    cmap='jet',
                    extend='neither')

for i, ax in enumerate(gs1_axs):
    for i_t in range(twoWT_25_0.wf.n_wt):
        # Add actuator discs
        draw_AD(ax,
                view='top',
                x=twoWT_25_0.wf.x_wt[i_t],
                y=twoWT_25_0.wf.y_wt[i_t],
                D=twoWT_25_0.wf.D/twoWT_25_0.wf.D,
                yaw=twoWT_25_0.wf.yaws[i_t])
    
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax.text(0.07, 0.9, subplot_labels[i] + ' ' + labels[i], transform=ax.transAxes + trans,
            fontsize='medium', va='bottom',  style='italic')

    # Set axes limits
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    
    # Set axes labels
    ax.set_ylabel('$y/D$')
gs1_axs[2].set_xlabel('$x/D$', labelpad=-1.5)

# Add colourbar
cax  = fig.add_axes([0.825, 0.4, 0.03, 0.47])
cbar = fig.colorbar(p, cax=cax)
cbar.set_label('$(V_{n} - V_{n-1})/U_{\infty,h}$')

for ip, x_D in enumerate(x_Ds):
    # Add lines to contour plots to show planes
    for ax in gs1_axs:
        ax.axvline(x_D, color='k', ls='-.')
        
    # Extract profile velocities
    V_pr = -twoWT_25_0.velocityProfile('Vdef', x_D, WT=1)
    V_pr_a = vel_pr_zh(twoWT_25_0, V_sur_zh, x_D, WT=0)
    V_pr_nwc = vel_pr_zh(twoWT_25_0, V_sur_zh_nwc, x_D, WT=0)
    
    # Plot profiles
    gs2_axs[ip].plot(V_pr/twoWT_25_0.wf.Uinf, twoWT_25_0.flowdata.y/twoWT_25_0.wf.D, color='k', label='RANS')
    gs2_axs[ip].plot(V_pr_a/U_h, twoWT_25_0.flowdata.y/twoWT_25_0.wf.D, color='r', label='Analytical')
    gs2_axs[ip].plot(V_pr_nwc/U_h, twoWT_25_0.flowdata.y/twoWT_25_0.wf.D, color='g', label='Corrected')
    
    # Add axes limits and labels
    gs2_axs[ip].set_ylim(ylims)
    gs2_axs[ip].set_xlabel('$\\frac{V_{n} - V_{n-1}}{U_{\infty,h}}$', fontsize=12)
    
    # Label subplots
    lab = r'$%sD$' % x_D
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    gs2_axs[ip].text(0.375, 0.925, subplot_labels[ip+2] + ' ' + lab, transform=gs2_axs[ip].transAxes + trans,
            fontsize='medium', va='bottom',  style='italic')

# Add y-label
gs2_axs[0].set_ylabel('$y/D$')

# Add legend
gs2_axs[-1].legend(loc='center left', bbox_to_anchor=(1.05, 0.5), frameon=False)

fig.savefig(fig_path + 'twoTurbine/V_zh_sur' + '.pdf', bbox_inches='tight')

plt.show()

# #%% Import flowcase
# twoWT_SWS = flowcase(dir_path = data_path + '2WT/SWS/',
#                  wf = copy(wf_2WT_25_0_template),
#                  cD = 8,
#                  shift=False)

# twoWT_SWS.flowdata = twoWT_SWS.flowdata.assign_coords(x=(twoWT_SWS.flowdata.x + 3.5*(twoWT_SWS.wf.n_wt-1)*twoWT_SWS.wf.D))

# #%% Plotting - lateral velocity surplus
# # fraction of textwidth
# fraction = 1

# # Axes limits
# xlims = (twoWT_SWS.wf.x_wt[0]-2, twoWT_SWS.wf.x_wt[-1]+8)
# ylims = (min(twoWT_SWS.wf.y_wt)-2, max(twoWT_SWS.wf.y_wt)+2)

# # Contourf limits
# cmin = -0.01
# cmax = 0.01

# # Downstream positions
# x_Ds = 7 + np.asarray([-1, 1, 3, 5, 7])

# # Subplot labels
# subplot_labels = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)']

# # Other labels
# labels = ['RANS', 'Analytical', 'Corrected']

# # Create figure and axes object
# fig = plt.figure(figsize=set_size(textwidth,
#                                   fraction,
#                                   height_adjust=1.33))

# # Make outer gridspec
# outer = gs.GridSpec(2, 1, figure=fig,
#               height_ratios=[3,1], hspace = 0.35, right=0.8)

# # Make nested gridspecs
# gs1 = gs.GridSpecFromSubplotSpec(3, 1,
#                                        subplot_spec = outer[0], hspace=0.4)
# gs2 = gs.GridSpecFromSubplotSpec(1, len(x_Ds),
#                                        subplot_spec = outer[1])

# # Create axes objects
# gs1_axs = []
# for grs in gs1:
#     gs1_axs.append(plt.subplot(grs))
# gs2_axs = []
# for grs in gs2:
#     gs2_axs.append(plt.subplot(grs))

# # First two share x and hide ticks
# gs1_axs[0].sharex(gs1_axs[2])
# gs1_axs[1].sharex(gs1_axs[2])
# plt.setp(gs1_axs[0].get_xticklabels(), visible=False)
# plt.setp(gs1_axs[1].get_xticklabels(), visible=False)

# # Share y on bottom row
# for ax in gs2_axs[1:]:
#     ax.sharex(gs2_axs[0])
#     plt.setp(ax.get_yticklabels(), visible=False)

# # PyWakeEllipSys results
# p = gs1_axs[0].contourf(twoWT_SWS.flowdata.x/twoWT_SWS.wf.D,
#                     twoWT_SWS.flowdata.y/twoWT_SWS.wf.D,
#                     twoWT_SWS.Vdef_zh[twoWT_SWS.wf.n_wt-1].T/U_h,
#                     levels=np.linspace(cmin,cmax,nlevels+1),
#                     vmin=cmin,
#                     vmax=cmax,
#                     cmap='jet',
#                     extend='neither')

# # Analytical results
# p = gs1_axs[1].contourf(twoWT_SWS.flowdata.x/twoWT_SWS.wf.D,
#                     twoWT_SWS.flowdata.y/twoWT_SWS.wf.D,
#                     V_sur_zh.T/U_h,
#                     levels=np.linspace(cmin,cmax,nlevels+1),
#                     vmin=cmin,
#                     vmax=cmax,
#                     cmap='jet',
#                     extend='neither')

# p = gs1_axs[2].contourf(twoWT_SWS.flowdata.x/twoWT_SWS.wf.D,
#                     twoWT_SWS.flowdata.y/twoWT_SWS.wf.D,
#                     V_sur_zh_nwc.T/U_h,
#                     levels=np.linspace(cmin,cmax,nlevels+1),
#                     vmin=cmin,
#                     vmax=cmax,
#                     cmap='jet',
#                     extend='neither')

# for i, ax in enumerate(gs1_axs):
#     for i_t in range(twoWT_SWS.wf.n_wt):
#         # Add actuator discs
#         draw_AD(ax,
#                 view='top',
#                 x=twoWT_SWS.wf.x_wt[i_t],
#                 y=twoWT_SWS.wf.y_wt[i_t],
#                 D=twoWT_SWS.wf.D/twoWT_SWS.wf.D,
#                 yaw=twoWT_SWS.wf.yaws[i_t])
    
#     trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
#     ax.text(0.07, 0.9, subplot_labels[i] + ' ' + labels[i], transform=ax.transAxes + trans,
#             fontsize='medium', va='bottom',  style='italic')

#     # Set axes limits
#     ax.set_xlim(xlims)
#     ax.set_ylim(ylims)
    
#     # Set axes labels
#     ax.set_ylabel('$y/D$')
# gs1_axs[2].set_xlabel('$x/D$', labelpad=-1.5)

# # Add colourbar
# cax  = fig.add_axes([0.825, 0.4, 0.03, 0.47])
# cbar = fig.colorbar(p, cax=cax)
# cbar.set_label('$(V_{n} - V_{n-1})/U_{\infty,h}$')

# for ip, x_D in enumerate(x_Ds):
#     # Add lines to contour plots to show planes
#     for ax in gs1_axs:
#         ax.axvline(x_D, color='k', ls='-.')
        
#     # Extract profile velocities
#     V_pr = -twoWT_SWS.velocityProfile('Vdef', x_D, WT=1)
#     V_pr_a = vel_pr_zh(twoWT_SWS, V_sur_zh, x_D, WT=0)
#     V_pr_nwc = vel_pr_zh(twoWT_SWS, V_sur_zh_nwc, x_D, WT=0)
    
#     # Plot profiles
#     gs2_axs[ip].plot(V_pr/twoWT_SWS.wf.Uinf, twoWT_SWS.flowdata.y/twoWT_SWS.wf.D, color='k', label='RANS')
#     gs2_axs[ip].plot(V_pr_a/U_h, twoWT_SWS.flowdata.y/twoWT_SWS.wf.D, color='r', label='Analytical')
#     gs2_axs[ip].plot(V_pr_nwc/U_h, twoWT_SWS.flowdata.y/twoWT_SWS.wf.D, color='g', label='Corrected')
    
#     # Add axes limits and labels
#     gs2_axs[ip].set_ylim(ylims)
#     gs2_axs[ip].set_xlabel('$\\frac{V_{n} - V_{n-1}}{U_{\infty,h}}$', fontsize=12)
    
#     # Label subplots
#     lab = r'$%sD$' % x_D
#     trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
#     gs2_axs[ip].text(0.375, 0.925, subplot_labels[ip+2] + ' ' + lab, transform=gs2_axs[ip].transAxes + trans,
#             fontsize='medium', va='bottom',  style='italic')

# # Add y-label
# gs2_axs[0].set_ylabel('$y/D$')

# # Add legend
# gs2_axs[-1].legend(loc='center left', bbox_to_anchor=(1.05, 0.5), frameon=False)

# fig.savefig(fig_path + 'twoTurbine/V_zh_SWS' + '.pdf', bbox_inches='tight')

# plt.show()

#%% Wind farm cases

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
WF_aligned = flowcase(dir_path=data_path + 'WF/aligned/on/',
                 wf=copy(wf_aligned),
                 cD=8)

WF_slanted = flowcase(dir_path=data_path + 'WF/slanted/on/',
                 wf=copy(wf_slanted),
                 cD=8)

#%% Shift turbine coordinates to match analytical results
WF_aligned.flowdata = WF_aligned.flowdata.assign_coords(x=(WF_aligned.flowdata.x + 3.5*(WF_aligned.wf.n_wt/3-1)*WF_aligned.wf.D))

WF_slanted.flowdata = WF_slanted.flowdata.assign_coords(x=(WF_slanted.flowdata.x + 14*WF_slanted.wf.D), y=(WF_slanted.flowdata.y + 1.5*WF_slanted.wf.D))

#%% Aligned wind farm
fc = WF_aligned
P = np.zeros((15, 4))

#%% PyWakeEllipSys power
P[:,0], _ = fc.powerAndThrust()

#%% Just streamwise analytical solution
flowdata_B, _, P[:,1], _ = fc.streamwiseSolution(method = 'original')

#%% Analytical solution
flowdata, _, P[:,2], _ = fc.analytical_solution(method = 'original',
                                           near_wake_correction=False)

flowdata_nwc, _, P[:,3], _ = fc.analytical_solution(method = 'original',
                                           near_wake_correction=True)

#%% Normalised power
P_norm = np.zeros(np.shape(P))
for i in range(np.shape(P)[1]):
    P_norm[:,i] = P[:,i]/P[1,0]
    if i > 0:
        P_norm[:,i] = P[:,i]/P[1,i]

#%% Hub height velocities
U_zh_B = flowdata_B.U.interp(z=fc.wf.zh)
U_zh, V_zh = vel_zh(flowdata, fc.wf.zh)
U_zh_nwc, V_zh_nwc = vel_zh(flowdata_nwc, fc.wf.zh)

#%% U_zh
# Fraction of textwidth
fraction = 1

# Axes limits
xlims = (-2, fc.wf.x_wt[-1]+8)
ylims = (min(fc.wf.y_wt)-3, max(fc.wf.y_wt)+3)

# Subplot labels
subplot_labels = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)']

# Other labels
labels = ['RANS', 'Bastankhah et al.', 'Analytical', 'Corrected']

# Colour limits
cmin = -0.1
cmax = 0.5

# Create figure and axes object
fig, axs = plt.subplots(4, 1, 
                        figsize=set_size(textwidth,
                                  fraction,
                                  height_adjust=2),
                        sharex=True)

# Plot contours of U_zh
p = axs[0].contourf(fc.flowdata.x/fc.wf.D,
                   fc.flowdata.y/fc.wf.D,
                   1-fc.U_zh.T/oneWT.wf.Uinf,
                   levels=np.linspace(cmin,cmax,nlevels+1),
                   vmin=cmin,
                   vmax=cmax,
                   cmap='jet')

p = axs[1].contourf(fc.flowdata.x/fc.wf.D,
                   fc.flowdata.y/fc.wf.D,
                   1-U_zh_B.T/U_h,
                   levels=np.linspace(cmin,cmax,nlevels+1),
                   vmin=cmin,
                   vmax=cmax,
                   cmap='jet')

p = axs[2].contourf(fc.flowdata.x/fc.wf.D,
                   fc.flowdata.y/fc.wf.D,
                   1-U_zh.T/U_h,
                   levels=np.linspace(cmin,cmax,nlevels+1),
                   vmin=cmin,
                   vmax=cmax,
                   cmap='jet')

p = axs[3].contourf(fc.flowdata.x/fc.wf.D,
                   fc.flowdata.y/fc.wf.D,
                   1-U_zh_nwc.T/U_h,
                   levels=np.linspace(cmin,cmax,nlevels+1),
                   vmin=cmin,
                   vmax=cmax,
                   cmap='jet')

for i, ax in enumerate(axs):
    for i_t in range(fc.wf.n_wt):
        # Add actuator discs
        draw_AD(ax,
                view='top',
                x=fc.wf.x_wt[i_t],
                y=fc.wf.y_wt[i_t],
                D=fc.wf.D/fc.wf.D,
                yaw=fc.wf.yaws[i_t])
        
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax.text(0.07, 0.94, subplot_labels[i] + ' ' + labels[i], transform=ax.transAxes + trans,
            fontsize='medium', va='bottom',  style='italic')
    
    # Set axes limits
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    
    # Set axes labels
    ax.set_ylabel('$y/D$')
axs[-1].set_xlabel('$x/D$')

# Add colourbar
plt.subplots_adjust(hspace=0.225, right=0.8)
cax  = fig.add_axes([0.825, 0.15, 0.03, 0.7])
cbar = fig.colorbar(p, cax=cax)
cbar.set_label('$1-(U/U_{\infty,h})$')

fig.savefig(fig_path + 'WF/aligned_Udef_zh' + '.pdf', bbox_inches='tight')

plt.show()

#%% Power-down-the-line
# Fraction of textwidth
fraction = 1

# Axes limits

# Create data
turbines = [1,2,3,4,5]
P_line   = P_norm[1::3,:] # middle row

# Create figure and axes objects
fig, ax = plt.subplots(1,1,
                       figsize=set_size(textwidth,
                                        fraction))

# Plot
ax.plot(turbines, P_line[:,0], color='k', marker='o', label='RANS')
ax.plot(turbines, P_line[:,1], color='b', marker='o', label='Bastankhah et al.')
ax.plot(turbines, P_line[:,2], color='r', marker='o', label='Analytical')
ax.plot(turbines, P_line[:,3], color='g', marker='o', label='Corrected')

# Axes labels
ax.set_xlabel('Turbine number')
ax.set_ylabel('$P/P_{1,RANS}$')

# Set x ticks
ax.set_xticks(turbines)

ax.legend()

fig.savefig(fig_path + 'WF/aligned_PDTL' + '.pdf', bbox_inches='tight')

plt.show()

#%% Total power
P_total_aligned = np.sum(P, axis=0)

#%% Slanted wind farm
fc = WF_slanted
P = np.zeros((15, 4))

#%% PyWakeEllipSys power
P[:,0], _ = fc.powerAndThrust()

#%% Just streamwise analytical solution
flowdata_B, _, P[:,1], _ = fc.streamwiseSolution(method = 'original')

#%% Analytical solution
flowdata, _, P[:,2], _ = fc.analytical_solution(method = 'original',
                                           near_wake_correction=False)

flowdata_nwc, _, P[:,3], _ = fc.analytical_solution(method = 'original',
                                           near_wake_correction=True)

#%% Normalised power
P_norm = np.zeros(np.shape(P))
for i in range(np.shape(P)[1]):
    P_norm[:,i] = P[:,i]/P[1,0]
    if i > 0:
        P_norm[:,i] = P[:,i]/P[1,i]

#%% Hub height velocities
U_zh_B = flowdata_B.U.interp(z=fc.wf.zh)
U_zh, V_zh = vel_zh(flowdata, fc.wf.zh)
U_zh_nwc, V_zh_nwc = vel_zh(flowdata_nwc, fc.wf.zh)

#%% U_zh
# Fraction of textwidth
fraction = 1

# Axes limits
xlims = (-2, fc.wf.x_wt[-1]+8)
ylims = (min(fc.wf.y_wt)-3, max(fc.wf.y_wt)+3)

# Subplot labels
subplot_labels = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)']

# Other labels
labels = ['RANS', 'Bastankhah et al.', 'Analytical', 'Corrected']

# Colour limits
cmin = -0.1
cmax = 0.5

# Create figure and axes object
fig, axs = plt.subplots(4, 1, 
                        figsize=set_size(textwidth,
                                  fraction,
                                  height_adjust=2),
                        sharex=True)

# Plot contours of U_zh
p = axs[0].contourf(fc.flowdata.x/fc.wf.D,
                   fc.flowdata.y/fc.wf.D,
                   1-fc.U_zh.T/oneWT.wf.Uinf,
                   levels=np.linspace(cmin,cmax,nlevels+1),
                   vmin=cmin,
                   vmax=cmax,
                   cmap='jet')

p = axs[1].contourf(fc.flowdata.x/fc.wf.D,
                   fc.flowdata.y/fc.wf.D,
                   1-U_zh_B.T/U_h,
                   levels=np.linspace(cmin,cmax,nlevels+1),
                   vmin=cmin,
                   vmax=cmax,
                   cmap='jet')

p = axs[2].contourf(fc.flowdata.x/fc.wf.D,
                   fc.flowdata.y/fc.wf.D,
                   1-U_zh.T/U_h,
                   levels=np.linspace(cmin,cmax,nlevels+1),
                   vmin=cmin,
                   vmax=cmax,
                   cmap='jet')

p = axs[3].contourf(fc.flowdata.x/fc.wf.D,
                   fc.flowdata.y/fc.wf.D,
                   1-U_zh_nwc.T/U_h,
                   levels=np.linspace(cmin,cmax,nlevels+1),
                   vmin=cmin,
                   vmax=cmax,
                   cmap='jet')

for i, ax in enumerate(axs):
    for i_t in range(fc.wf.n_wt):
        # Add actuator discs
        draw_AD(ax,
                view='top',
                x=fc.wf.x_wt[i_t],
                y=fc.wf.y_wt[i_t],
                D=fc.wf.D/fc.wf.D,
                yaw=fc.wf.yaws[i_t])
        
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax.text(0.07, 0.94, subplot_labels[i] + ' ' + labels[i], transform=ax.transAxes + trans,
            fontsize='medium', va='bottom',  style='italic')
    
    # Set axes limits
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    
    # Set axes labels
    ax.set_ylabel('$y/D$')
axs[-1].set_xlabel('$x/D$')

# Add colourbar
plt.subplots_adjust(hspace=0.225, right=0.8)
cax  = fig.add_axes([0.825, 0.15, 0.03, 0.7])
cbar = fig.colorbar(p, cax=cax)
cbar.set_label('$1-(U/U_{\infty,h})$')

fig.savefig(fig_path + 'WF/slanted_Udef_zh' + '.pdf', bbox_inches='tight')

plt.show()

#%% Power-down-the-line
# Fraction of textwidth
fraction = 1

# Axes limits

# Create data
turbines = [1,2,3,4,5]
P_line   = P_norm[1::3,:] # middle row

# Create figure and axes objects
fig, ax = plt.subplots(1,1,
                       figsize=set_size(textwidth,
                                        fraction))

# Plot
ax.plot(turbines, P_line[:,0], color='k', marker='o', label='RANS')
ax.plot(turbines, P_line[:,1], color='b', marker='o', label='Bastankhah et al.')
ax.plot(turbines, P_line[:,2], color='r', marker='o', label='Analytical')
ax.plot(turbines, P_line[:,3], color='g', marker='o', label='Corrected')

# Axes labels
ax.set_xlabel('Turbine number')
ax.set_ylabel('$P/P_{1,RANS}$')

# Set x ticks
ax.set_xticks(turbines)

ax.legend()

fig.savefig(fig_path + 'WF/slanted_PDTL' + '.pdf', bbox_inches='tight')

plt.show()

#%% Total power
P_total_slanted = np.sum(P, axis=0)