# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 11:33:36 2022

@author: nilsg
"""

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D
from pylab import cm
import numpy as np
import os
import copy

# Edit the font, font size, and axes width
plt.rcParams.update({
  "text.usetex": True,
  "font.family": "Tex Gyre Adventor"
})
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1

os.chdir('C:/Users/nilsg/OneDrive/Documents/EWEM/Thesis/NUMERICAL/PyWakeEllipSys/')

n_x_D = 50 # discretisation of x/D for wake width and centre plots

import post_utils as pu
from post_utils import windTurbine, windFarm, flowcase, set_size

NREL5MW = windTurbine(D=126.0, zh=90.0, TSR=7.5)

wf_template = windFarm(Uinf=8,
              ti=0.06,
              x_wt=[0],
              y_wt=[0],
              wts=[NREL5MW],
              yaws=[25],
              CTs=[0.8],
              wrs=[True])

wf_aligned = windFarm(Uinf=8,
                      ti=0.06,
                      x_wt=[ 0,  0,  0,
                             7,  7,  7,
                            14, 14, 14,
                            21, 21, 21,
                            28, 28, 28],
                      y_wt=[ 8,  0,  4,
                             8,  0,  4,
                             8,  0,  4,
                             8,  0,  4,
                             8,  0,  4],
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
                      y_wt=[ 8.00,  0.00,  4.00,
                             8.75,  0.75,  4.75,
                             9.50,  1.50,  5.50,
                            10.25,  2.25,  6.25,
                            11.00,  3.00,  7.00],
                      wts=[NREL5MW]*15,
                      yaws=[25]*15,
                      CTs=[0.8]*15,
                      wrs=[True]*15)

data_path = 'C:/Users/nilsg/OneDrive/Documents/EWEM/Thesis/NUMERICAL/PyWakeEllipSys/flowdata/LSS/'

fig_path = 'C:/Users/nilsg/Dropbox/Apps/Overleaf/Thesis/figures/03_lateralWake/'

fig_path_derivation = 'C:/Users/nilsg/Dropbox/Apps/Overleaf/Thesis/figures/04_derivation/'

textwidth = 448.0 # [pts]

#%% Create flowcases
# Downstream distance
yaw25 = flowcase(dir_path=data_path + 'generalSolution/standalone/',
                 wf=copy.copy(wf_template),
                 cD=8)
yaw25.wf.yaws = [25]

# Zero yaw
yaw0 = flowcase(dir_path=data_path + 'appendix/yaw/0/',
                wf=copy.copy(wf_template),
                cD=8)
yaw0.wf.yaws = [0]

# Yaw angles
yaws = [40,30,20,10,0,-10,-20,-30,-40]
fc_yaws = []
for iy, yaw in enumerate(yaws):
    fc_yaws.append(flowcase(dir_path=data_path + 'appendix/yaw/' + str(yaw) + '/',
                            wf=copy.copy(wf_template),
                            cD=8))
    fc_yaws[iy].wf.yaws = [yaw]
    
# Yaw angles (long domain)
yaws_LD = [40,30,20,10,0]
fc_yaws_LD = []
for iy_LD, yaw_LD in enumerate(yaws_LD):
    fc_yaws_LD.append(flowcase(dir_path=data_path + 'appendix/yaw/long_domain/' + str(yaw_LD) + '/',
                            wf=copy.copy(wf_template),
                            cD=8))
    fc_yaws_LD[iy_LD].wf.yaws = [yaw_LD]
    
# Thrust coefficients
CTs = [0.8, 0.6, 0.4, 0.2]

fc_CTs = []
for iCT, CT in enumerate(CTs):
    fc_CTs.append(flowcase(dir_path=data_path + 'appendix/ct/' + str(CT) + '/',
                            wf=copy.copy(wf_template),
                            cD=8))
    fc_CTs[iCT].wf.CTs = [CT]

# Turbulence intensities
TIs = [0.10, 0.08, 0.06, 0.04]

fc_TIs = []
for iTI, TI in enumerate(TIs):
    fc_TIs.append(flowcase(dir_path=data_path + 'appendix/ti/' + str(TI) + '/',
                            wf=copy.copy(wf_template),
                            cD=8))
    fc_TIs[iTI].wf.ti = TI

fc_yaws_nwr = []
for iy, yaw in enumerate(yaws):
    fc_yaws_nwr.append(flowcase(dir_path=data_path + 'appendix/nwr/' + str(yaw) + '/',
                            wf=copy.copy(wf_template),
                            cD=8))
    fc_yaws_nwr[iy].wf.yaws = [yaw]
    fc_yaws_nwr[iy].wf.wrs = [False]

fc_SS = [yaw25]
fc_SS.append(flowcase(dir_path=data_path + 'generalSolution/aligned/',
                      wf=copy.copy(wf_aligned),
                      cD=8))
fc_SS.append(flowcase(dir_path=data_path + 'generalSolution/slanted/',
                      wf=copy.copy(wf_slanted),
                      cD=8))

#%% Velocity profile: Effect of downstream distance: yaw25
# Fraction of textwidth
fraction = 1

# Axes limits
xlims = (-2, 8)
ylims = (-3, 3)

# Downstream positions
x_Ds = [-1,1,1.5,3,5,7]

# Plot contourf and profiles
fig, axs = plt.subplots(2, len(x_Ds), figsize=set_size(textwidth, fraction=fraction), sharey=True)
gs = axs[0,0].get_gridspec() # get geometry of subplot grid

# Remove top row of subplot
for ax in axs[0,:]:
    ax.remove()
    
# Replace with a single plot spanning the whole width
axbig = fig.add_subplot(gs[0,:])

# Plot contour of lateral velocity at hub height
cmin = -0.10
cmax = 0.10
p = axbig.contourf(yaw25.V_zh.x/yaw25.wf.D,
                   yaw25.V_zh.y/yaw25.wf.D,
                   yaw25.V_zh.T/8,
                   levels=np.linspace(cmin,cmax,11), cmap='jet',
                   vmin=cmin, vmax=cmax,
                   extend='both')

# Add actuator disc to plot
for i_wt in range(len(yaw25.wf.wts)):
    pu.draw_AD(ax=axbig,
               view='top',
               x=yaw25.wf.x_AD[i_wt],
               y=yaw25.wf.y_AD[i_wt],
               D=yaw25.wf.D/yaw25.wf.D,
               yaw=yaw25.wf.yaws[i_wt-1])
    
trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
axbig.text(0.05, 0.95, 'a)', transform=axbig.transAxes + trans, fontsize='medium', va='bottom', style='italic')

# Set axes limits
axbig.set_xlim(xlims)
axbig.set_ylim(ylims)

# Set axes ticks
# axbig.set_xticks([0,1,2,3,4,5,6,7,8])
axbig.set_yticks([-2,0,2])

# Set axes labels
axbig.set_ylabel('$y/D$')
axbig.set_xlabel('$x/D$', labelpad=-2)

# Set tick params
axbig.xaxis.set_tick_params(which='major', size=5, width=1, direction='in', top='on')
axbig.yaxis.set_tick_params(which='major', size=5, width=1, direction='in', right='on')

# Add colourbar
divider = make_axes_locatable(axbig)
cax     = divider.append_axes('right',
                             size='3%',
                             pad=0.075)
cbar    = fig.colorbar(p, ax=axbig, cax=cax)
cbar.set_label('$V/U_{\infty,h}$')

subplot_labels = ['b)', 'c)', 'd)', 'e)', 'f)', 'g)']

for ip, x_D in enumerate(x_Ds):
    # Add lines to contour plot to show planes
    axbig.vlines(x_D, ylims[0], ylims[1], color='k', ls='-.')
    
    # Extract profile velocities
    V_pr = yaw25.velocityProfile('V', x_D)
    
    # Plot profiles
    # axs[1,ip].scatter(V_pr,
    #                   pu.y_to_ys(V_pr.y, V_pr, 'Gaussian')/yaw25.wf.D,
    #                   s=5, c='k')
    axs[1,ip].plot(V_pr/8,
                   V_pr.y/yaw25.wf.D,
                   c='k')
    
    # Add axes limits and labels
    axs[1,ip].set_ylim(ylims)
    axs[1,ip].set_yticks([-2,0,2])
    axs[1,ip].set_xlabel('$V/U_{\infty,h}$')
    
    # Label each plot with x/D
    lab = r'$%sD$' % x_D
    # axs[1,ip].text(0.98, 0.95,
    #                lab,
    #                horizontalalignment='right',
    #                verticalalignment='top',
    #                transform=axs[1,ip].transAxes)
    
    # Label subplots
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    axs[1,ip].text(0.35, 0.95, subplot_labels[ip] + ' ' + lab, transform=axs[1,ip].transAxes + trans,
            fontsize='medium', va='bottom',  style='italic')
    
    axs[1,ip].set_xlim(-0.03, 0.09)
    axs[1,ip].set_xticks([0, 0.05], ['0', '0.05'])

axs[1,0].set_ylabel('$y/D$')

plt.subplots_adjust(wspace=0.2, 
                    hspace=0.5)

fig.savefig(fig_path + 'velocityProfile/x.pdf', bbox_inches='tight')

plt.show()

#%% Velocity profile: Effect of downstream distance: zero yaw
# Fraction of textwidth
fraction = 1

# Axes limits
xlims = (-2, 8)
ylims = (-3, 3)

# Downstream positions
x_Ds = [-1,1,1.5,3,5]

# Plot contourf and profiles
fig, axs = plt.subplots(2, len(x_Ds), figsize=set_size(textwidth, fraction=fraction), sharey=True)
gs = axs[0,0].get_gridspec() # get geometry of subplot grid

# Remove top row of subplot
for ax in axs[0,:]:
    ax.remove()
    
# Replace with a single plot spanning the whole width
axbig = fig.add_subplot(gs[0,:])

# Plot contour of lateral velocity at hub height
cmin = -0.10
cmax = 0.10
p = axbig.contourf(yaw0.V_zh.x/yaw0.wf.D,
                   yaw0.V_zh.y/yaw0.wf.D,
                   yaw0.V_zh.T/8,
                   levels=np.linspace(cmin,cmax,11), cmap='jet',
                   vmin=cmin, vmax=cmax,
                   extend='both')

# Add actuator disc to plot
for i_wt in range(len(yaw25.wf.wts)):
    pu.draw_AD(ax=axbig,
               view='top',
               x=yaw0.wf.x_AD[i_wt],
               y=yaw0.wf.y_AD[i_wt],
               D=yaw0.wf.D/yaw0.wf.D,
               yaw=yaw0.wf.yaws[i_wt-1])
    
trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
axbig.text(0.05, 0.95, 'a)', transform=axbig.transAxes + trans, fontsize='medium', va='bottom', style='italic')

# Set axes limits
axbig.set_xlim(xlims)
axbig.set_ylim(ylims)

# Set axes ticks
# axbig.set_xticks([0,1,2,3,4,5,6])
axbig.set_yticks([-2,0,2])

# Set axes labels
axbig.set_ylabel('$y/D$')
axbig.set_xlabel('$x/D$', labelpad=0)

# Set tick params
axbig.xaxis.set_tick_params(which='major', size=5, width=1, direction='in', top='on')
axbig.yaxis.set_tick_params(which='major', size=5, width=1, direction='in', right='on')

# Add colourbar
divider = make_axes_locatable(axbig)
cax     = divider.append_axes('right',
                             size='3%',
                             pad=0.075)
cbar    = fig.colorbar(p, ax=axbig, cax=cax)
cbar.set_label('$V/U_{\infty,h}$')

subplot_labels = ['b)', 'c)', 'd)', 'e)', 'f)', 'g)']

for ip, x_D in enumerate(x_Ds):
    # Add lines to contour plot to show planes
    axbig.vlines(x_D, ylims[0], ylims[1], color='k', ls='-.')
    
    # Extract profile velocities
    V_pr = yaw0.velocityProfile('V', x_D)
    
    # Plot profiles
    # axs[1,ip].plot(V_pr,
    #                   pu.y_to_ys(V_pr.y, V_pr, 'Gaussian')/yaw25.wf.D,
    #                   c='k')
    axs[1,ip].plot(V_pr/8,
                   V_pr.y/yaw0.wf.D,
                   c='k')
    
    # Add axes limits and labels
    axs[1,ip].set_ylim(ylims)
    axs[1,ip].set_yticks([-2,0,2])
    axs[1,ip].set_xlabel('$V/U_{\infty,h}$')
    
    # Label each plot with x/D
    lab = r'$%sD$' % str(x_D)
    # axs[1,ip].text(0.98, 0.95,
    #                lab,
    #                horizontalalignment='right',
    #                verticalalignment='top',
    #                transform=axs[1,ip].transAxes)
    
    # Label subplots
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    axs[1,ip].text(0.35, 0.95, subplot_labels[ip] + ' ' + lab, transform=axs[1,ip].transAxes + trans,
            fontsize='medium', va='bottom',  style='italic')
    
    axs[1,ip].set_xlim(-0.023, 0.023)
    axs[1,ip].set_xticks([-0.02, 0, 0.02], ['-0.02', '0', '0.02'])

axs[1,0].set_ylabel('$y/D$')

plt.subplots_adjust(wspace=0.5, 
                    hspace=0.55)

fig.savefig(fig_path + 'velocityProfile/x_yaw0.pdf', bbox_inches='tight')

plt.show()

#%% Velocity profile: Effect of downstream distance: streamlines: x-y plane

subplot_labels = ['a)','b)']

fig, axs = plt.subplots(2,1,figsize=set_size(textwidth, fraction=1), sharex=True)

for i, fc in enumerate([yaw0, yaw25]):
    fc.flowdata_r = fc.flowdata.where((fc.flowdata.x > -5*fc.wf.D) & (fc.flowdata.x < 10*fc.wf.D), drop=True)
    fc.flowdata_r = fc.flowdata_r.where((fc.flowdata.y > -3*fc.wf.D) & (fc.flowdata.y < 3*fc.wf.D), drop=True)
    
    X, Y = np.meshgrid(fc.flowdata_r.x/fc.wf.D,
                       fc.flowdata_r.y/fc.wf.D)
    
    U_zh = fc.flowdata_r.U.interp(z=90)
    V_zh = fc.flowdata_r.V.interp(z=90)
    
    cmin = -0.1
    cmax = 0.1
    p = axs[i].contourf(X, Y, V_zh.T/8, cmap='jet', levels=np.linspace(cmin,cmax,11), vmin=cmin, vmax=cmax, extend='both')
    axs[i].streamplot(X, Y, U_zh.T, V_zh.T, color='k', density=0.5)
    axs[i].set_xlim(-2,None)
    axs[i].set_ylim(-1.5,1.5)
    axs[i].set_ylabel('$y/D$')
    
    pu.draw_AD(axs[i], 'top', 0, 0, fc.wf.D/yaw0.wf.D, fc.wf.yaws[0])
    
    # Label subplots
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    axs[i].text(0.07, 0.95, subplot_labels[i] + ' $\gamma = {:d}^\circ$'.format(fc.wf.yaws[0]), transform=axs[i].transAxes + trans,
            fontsize='medium', va='bottom',  style='italic')
    
axs[1].set_xlabel('$x/D$')

plt.subplots_adjust(right=0.8,
                    hspace=0.3)

cax = fig.add_axes([0.825, 0.15, 0.03, 0.7])
cbar = fig.colorbar(p, cax=cax)
cbar.set_label('$V/U_{\infty,h}$')

fig.savefig(fig_path + 'velocityProfile/x_streamlines_zh.pdf', bbox_inches='tight')

plt.show()

#%% Velocity profile: Effect of downstream distance: streamlines: x-z plane

subplot_labels = ['a)','b)']

fig, axs = plt.subplots(2,1,figsize=set_size(textwidth, fraction=1), sharex=True)

for i, fc in enumerate([yaw0, yaw25]):
    fc.flowdata_r = fc.flowdata.where((fc.flowdata.x > -5*fc.wf.D) & (fc.flowdata.x < 11*fc.wf.D), drop=True)
    # fc.flowdata_r = fc.flowdata_r.where(fc.flowdata.x < 3*fc.wf.zh, drop=True)
    
    fc.z_linear = np.linspace(0, 3*fc.wf.zh, 1000)
    
    X, Z = np.meshgrid(fc.flowdata_r.x/fc.wf.D,
                       fc.z_linear/fc.wf.zh)
    
    U_yh = fc.flowdata_r.U.interp(y=0, z=fc.z_linear)
    V_yh = fc.flowdata_r.V.interp(y=0, z=fc.z_linear)
    W_yh = fc.flowdata_r.W.interp(y=0, z=fc.z_linear)
    
    cmin = -0.1
    cmax = 0.1
    p = axs[i].contourf(X, Z, V_yh.T/8, cmap='jet', levels=np.linspace(cmin,cmax,11), vmin=cmin, vmax=cmax, extend='both')
    axs[i].streamplot(X, Z, U_yh.T, W_yh.T, color='k', density=0.5)
    axs[i].set_xlim(-2,10)
    axs[i].set_ylim(0,1.95)
    axs[i].set_ylabel('$z/z_h$')
    
    pu.draw_AD(axs[i], 'side', 0, 1, fc.wf.D/yaw0.wf.D, fc.wf.yaws[0])
    
    # Label subplots
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    axs[i].text(0.07, 0.95, subplot_labels[i] + ' $\gamma = {:d}^\circ$'.format(fc.wf.yaws[0]), transform=axs[i].transAxes + trans,
            fontsize='medium', va='bottom',  style='italic')
    
axs[1].set_xlabel('$x/D$')

plt.subplots_adjust(right=0.8,
                    hspace=0.3)

cax = fig.add_axes([0.825, 0.15, 0.03, 0.7])
cbar = fig.colorbar(p, cax=cax)
cbar.set_label('$V/U_{\infty,h}$')

fig.savefig(fig_path + 'velocityProfile/x_streamlines_yh.pdf', bbox_inches='tight')

plt.show()
    
#%% Effect of yaw angle
# Fraction of textwidth
fraction = 1

# Downstream position
x_D = 6

# Axes limits
xlims = (-3,3)

# Generate 2 colors from the 'tab10' colormap
colours = cm.get_cmap('jet', 9)

# Create markers
markers = ["o", "s", "D", "P", "X", "^", "v", "<", ">"]

# Create figure and axes
fig, ax = plt.subplots(1,1,
                       figsize=set_size(textwidth,fraction))

for i, fc in enumerate(fc_yaws):
    # Extract profile
    V_pr = fc.velocityProfile('V', x_D)
    
    # Plot profile with label
    # ax.scatter(V_pr.y/fc.wf.D, V_pr/8, label='$' + str(fc.wf.yaws[0]) + '^\circ$', color=colours(i), marker=markers[i])
    ax.plot(V_pr.y/fc.wf.D, V_pr/8, label='$' + str(fc.wf.yaws[0]) + '^\circ$', color=colours(i), linewidth=2)
    
# Add axes limits and labels
ax.set_xlim(xlims)
ax.set_xlabel('$y/D$')
ax.set_ylabel('$V/U_{\infty,h}$')

# Add legend outside axes
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='$\gamma$', frameon=False)

fig.savefig(fig_path + 'velocityProfile/yaw.pdf', bbox_inches='tight')

plt.show()

#%% Effect of thrust coefficient
# Fraction of textwidth
fraction = 0.8

# Downstream position
x_D = 6

# Axes limits
xlims = (-3,3)

# Generate 2 colors from the 'tab10' colormap
colours = cm.get_cmap('jet', 9)

# Create markers
markers = ["o", "s", "D", "P", "X", "^", "v", "<", ">"]

# Create figure and axes
fig, ax = plt.subplots(1,1,
                       figsize=set_size(textwidth,fraction))

for i, fc in enumerate(fc_CTs):
    # Extract profile
    V_pr = fc.velocityProfile('V', x_D)
    
    # Plot profile with label
    # ax.scatter(V_pr.y/fc.wf.D, V_pr, label='$' + str(fc.wf.CTs[0]) + '$', color=colours(i), marker=markers[i])
    ax.plot(V_pr.y/fc.wf.D, V_pr/8, label='$' + str(fc.wf.CTs[0]) + '$', color=colours(i), linewidth=2)
    
# Add axes limits and labels
ax.set_xlim(xlims)
ax.set_xlabel('$y/D$')
ax.set_ylabel('$V/U_{\infty,h}$')

# Add legend outside axes
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
ax.legend(loc='upper right', title='$C_T$', frameon=False)

fig.savefig(fig_path + 'velocityProfile/CT.pdf', bbox_inches='tight')

plt.show()

#%% Effect of thrust coefficient
# Fraction of textwidth
fraction = 0.8

# Downstream position
x_D = 6

# Axes limits
xlims = (-3,3)

# Generate 2 colors from the 'tab10' colormap
colours = cm.get_cmap('jet', 9)

# Create markers
markers = ["o", "s", "D", "P", "X", "^", "v", "<", ">"]

# Create figure and axes
fig, ax = plt.subplots(1,1,
                       figsize=set_size(textwidth,fraction))

for i, fc in enumerate(fc_CTs):
    # Extract profile
    V_pr = fc.velocityProfile('V', x_D)
    
    # Plot profile with label
    # ax.scatter(V_pr.y/fc.wf.D, V_pr/(8*fc.wf.CTs[0]), label='$' + str(fc.wf.CTs[0]) + '$', color=colours(i), marker=markers[i])
    ax.plot(V_pr.y/fc.wf.D, V_pr/(8*fc.wf.CTs[0]), label='$' + str(fc.wf.CTs[0]) + '$', color=colours(i), linewidth=2)

# Add axes limits and labels
ax.set_xlim(xlims)
ax.set_xlabel('$y/D$')
ax.set_ylabel('$V/(U_{\infty,h}C_T)$')

# Add legend outside axes
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
ax.legend(loc='upper right', title='$C_T$', frameon=False)

fig.savefig(fig_path + 'velocityProfile/CT_norm.pdf', bbox_inches='tight')

plt.show()

#%% Effect of turbulence intensity
# Fraction of textwidth
fraction = 0.8

# Downstream position
x_D = 6

# Axes limits
xlims = (-3,3)

# Generate 2 colors from the 'tab10' colormap
colours = cm.get_cmap('jet', 9)

# Create markers
markers = ["o", "s", "D", "P", "X", "^", "v", "<", ">"]

# Create figure and axes
fig, ax = plt.subplots(1,1,
                       figsize=set_size(textwidth,fraction))

for i, fc in enumerate(fc_TIs):
    # Extract profile
    V_pr = fc.velocityProfile('V', x_D)
    
    # Plot profile with label
    # ax.scatter(V_pr.y/fc.wf.D, V_pr/8, label='$' + '{:.2f}'.format(fc.wf.ti) + '$', color=colours(i), marker=markers[i])
    ax.plot(V_pr.y/fc.wf.D, V_pr/8, label='$' + '{:.2f}'.format(fc.wf.ti) + '$', color=colours(i), linewidth=2)
    
# Add axes limits and labels
ax.set_xlim(xlims)
ax.set_xlabel('$y/D$')
ax.set_ylabel('$V/U_{\infty,h}$')

# Add legend outside axes
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
ax.legend(loc='upper right', title='$I_{\infty,h}$', frameon=False)

fig.savefig(fig_path + 'velocityProfile/TI.pdf', bbox_inches='tight')

plt.show()

#%% Effect of wake rotation
# Fraction of textwidth
fraction = 0.8

# Downstream position
x_D = 6

# Axes limits
xlims = (-3,3)

# Generate 2 colors from the 'tab10' colormap
colours = cm.get_cmap('jet', 9)

# Create markers
markers = ["o", "s", "D", "P", "X", "^", "v", "<", ">"]

# Create figure and axes
fig, ax = plt.subplots(1,1,
                       figsize=set_size(textwidth,fraction))

# Create empty list for legend handles
wr_plot = []

for iy in range(len(yaws[:-4])):
    # Extract profiles
    V_pr_wr  = fc_yaws[iy].velocityProfile('V', x_D)
    V_pr_nwr = fc_yaws_nwr[iy].velocityProfile('V', x_D)
    
    # Plot profiles with labels
    l1, = ax.plot(V_pr_wr.y/fc.wf.D, 
               V_pr_wr/8, 
               label='$' + str(fc_yaws[iy].wf.yaws[0]) + '^\circ$', 
               color=colours(iy),
               ls='-',
               linewidth=2)
    wr_plot.append(l1)
    ax.plot(V_pr_nwr.y/fc.wf.D, 
               V_pr_nwr/8,
               color=colours(iy),
               ls='--',
               linewidth=2)
    
# Add axes limits and labels
ax.set_xlim(xlims)
ax.set_xlabel('$y/D$')
ax.set_ylabel('$V/U_{\infty,h}$')

# Create legend entries for line styles
solid_line = Line2D([], [], 
                       color=colours(0),
                       ls='-',
                       label='on')
dashed_line = Line2D([], [], 
                       color=colours(0),
                       ls='--',
                       label='off')

# Add legend outside axes
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
yaw_legend = ax.legend(handles=wr_plot, 
                       loc='upper right',
                       title='$\gamma$',
                       frameon=False)
plt.gca().add_artist(yaw_legend)
ax.legend(handles=[solid_line, dashed_line],
          loc='upper left',
          title='Wake rotation',
          frameon=False)

fig.savefig(fig_path + 'velocityProfile/wr.pdf', bbox_inches='tight')

plt.show()

#%% Self similarity: Effect of yaw angle
# fraction of textwidth
fraction = 1

# Wake centre and width methods
wcm = 'max'; wwm = 'Gaussian'

# Downstream positions
x_Ds = [3,4,5,6,7,8]

# Axes limits
xlims=(-4,4)

# Create markers
markers = ["o", "s", "D", "P", "X", "^", "v", "<", ">"]

# Generate 2 colors from the 'tab10' colormap
colours = cm.get_cmap('jet', len(x_Ds))

# Create subplot labels
subplot_labels = ['a)', 'b)', 'c)', 'd)']

# Create subplots object
fig, axs = plt.subplots(2,2,
                        figsize=set_size(textwidth, fraction), sharex=True)

# Make axes indexing 1D
axs = np.ravel(axs)

for iy, fc in enumerate(reversed(fc_yaws[1:5])):
    for ip, x_D in enumerate(x_Ds):
        # Extract profile
        V_pr = fc.velocityProfile('V', x_D)
        
        # Calculate wake centre velocity
        if fc.wf.yaws[0] > 0:
            _, V_c = pu.wakeCentre(wcm, V_pr.y, V_pr)
        else:
            V_c = np.max(np.abs(V_pr.to_numpy()))
            
        # Calculate wake width
        sigma = pu.wakeWidth(wwm, V_pr.y, V_pr)
        
        # Plot profile with label
        lab = '${:d}$'.format(x_D)
        if fc.wf.yaws[0] == 0:
            # axs[iy].scatter(V_pr.y/sigma, V_pr/V_c, label=lab, color=colours(ip), marker=markers[ip])
            axs[iy].plot(V_pr.y/sigma, V_pr/V_c, label=lab, color=colours(ip), linewidth=2)
        else:
            # axs[iy].scatter(pu.y_to_ys(V_pr.y, V_pr, wcm)/sigma, V_pr/V_c, label=lab, color=colours(ip), marker=markers[ip])
            axs[iy].plot(pu.y_to_ys(V_pr.y, V_pr, wcm)/sigma, V_pr/V_c, label=lab, color=colours(ip), linewidth=2)
    
    # Add axes limits and labels
    axs[iy].set_xlim(xlims)
    
    if iy > 1:
        axs[iy].set_xlabel('$y^{*,2}/\sigma_{y,2}$')
    if iy%2 == 0:
        axs[iy].set_ylabel('$V/V_c$')
        
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    axs[iy].text(0.12, 0.95, 
                 subplot_labels[iy] + ' $\gamma = {:d}^\circ$'.format(fc.wf.yaws[0]),
                 transform=axs[iy].transAxes + trans,
            fontsize='medium', va='bottom',  style='italic')
    
plt.legend(loc='center left', bbox_to_anchor=(1.025,1.1), title='$x/D$', frameon=False)

plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.18, 
                    hspace=0.25)

fig.savefig(fig_path + 'selfSimilarity/yaw.pdf', bbox_inches='tight')

plt.show()

#%% Self similarity: Effect of thrust coefficient
# fraction of textwidth
fraction = 1

# Wake centre and width methods
wcm = 'max'; wwm = 'integral'

# Downstream positions
x_Ds = [3,4,5,6,7,8]

# Axes limits
xlims=(-4,4)
ylims=(None,1.1)

# Create markers
markers = ["o", "s", "D", "P", "X", "^", "v", "<", ">"]

# Generate 2 colors from the 'tab10' colormap
colours = cm.get_cmap('jet', len(x_Ds))

# Create subplot labels
subplot_labels = ['a)', 'b)', 'c)', 'd)']

# Create subplots object
fig, axs = plt.subplots(2,2,
                        figsize=set_size(textwidth, fraction), sharex=True, sharey=True)

# Make axes indexing 1D
axs = np.ravel(axs)

for iy, fc in enumerate(reversed(fc_CTs)):
    for ip, x_D in enumerate(x_Ds):
        # Extract profile
        V_pr = fc.velocityProfile('V', x_D)
        
        # Calculate wake centre velocity
        if fc.wf.yaws[0] > 0:
            _, V_c = pu.wakeCentre(wcm, V_pr.y, V_pr)
        else:
            V_c = np.max(np.abs(V_pr.to_numpy()))
            
        # Calculate wake width
        sigma = pu.wakeWidth(wwm, V_pr.y, V_pr)
        
        # Plot profile with label
        lab = '${:d}$'.format(x_D)
        if fc.wf.yaws[0] == 0:
            # axs[iy].scatter(V_pr.y/sigma, V_pr/V_c, label=lab, color=colours(ip), marker=markers[ip])
            axs[iy].plot(V_pr.y/sigma, V_pr/V_c, label=lab, color=colours(ip), linewidth=2)
        else:
            # axs[iy].scatter(pu.y_to_ys(V_pr.y, V_pr, wcm)/sigma, V_pr/V_c, label=lab, color=colours(ip), marker=markers[ip])
            axs[iy].plot(pu.y_to_ys(V_pr.y, V_pr, wcm)/sigma, V_pr/V_c, label=lab, color=colours(ip), linewidth=2)
    
    # Add axes limits and labels
    axs[iy].set_xlim(xlims)
    axs[iy].set_ylim(ylims)
    
    if iy > 1:
        axs[iy].set_xlabel('$y^{*,2}/\sigma_{y,2}$')
    if iy%2 == 0:
        axs[iy].set_ylabel('$V/V_c$')
        
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    axs[iy].text(0.12, 0.94, 
                 subplot_labels[iy] + ' $C_T = {:.1f}$'.format(fc.wf.CTs[0]),
                 transform=axs[iy].transAxes + trans,
            fontsize='medium', va='bottom',  style='italic')
    
plt.legend(loc='center left', bbox_to_anchor=(1.025,1.1), title='$x/D$', frameon=False)

plt.subplots_adjust(wspace=0.18, 
                    hspace=0.25)

fig.savefig(fig_path + 'selfSimilarity/CT.pdf', bbox_inches='tight')

plt.show()

#%% Self-similarity: effect of thrust coefficient: one figure
# fraction of textwidth
fraction = 1

# Wake centre and width methods
wcm = 'max'; wwm = 'integral'

# Downstream positions
x_Ds = [3,4,5,6,7,8]

# Axes limits
xlims=(-4,4)

# Create markers
markers = ["o", "s", "D", "P", "X", "^", "v", "<", ">"]

# Generate 2 colors from the 'tab10' colormap
colours = cm.get_cmap('jet', len(x_Ds))

# Create subplot labels
subplot_labels = ['a)', 'b)', 'c)', 'd)']

# Create subplots object
fig, ax = plt.subplots(1,1,
                        figsize=set_size(textwidth, fraction))

for iy, fc in enumerate(reversed(fc_CTs)):
    handles = []
    for ip, x_D in enumerate(x_Ds):
        # Extract profile
        V_pr = fc.velocityProfile('V', x_D)
        
        # Calculate wake centre velocity
        if fc.wf.yaws[0] > 0:
            _, V_c = pu.wakeCentre(wcm, V_pr.y, V_pr)
        else:
            V_c = np.max(np.abs(V_pr.to_numpy()))
            
        # Calculate wake width
        sigma = pu.wakeWidth(wwm, V_pr.y, V_pr)
        
        # Plot profile with label
        lab = '${:d}$'.format(x_D)
        if fc.wf.yaws[0] == 0:
            # ax.scatter(V_pr.y/sigma, V_pr/V_c, label=lab, color=colours(ip), marker=markers[ip])
            ax.plot(V_pr.y/sigma, V_pr/V_c, label=lab, color=colours(ip), linewidth=2)
        else:
            # l1 = ax.scatter(pu.y_to_ys(V_pr.y, V_pr, wcm)/sigma, V_pr/V_c, label=lab, color=colours(ip), marker=markers[ip])
            l1, = ax.plot(pu.y_to_ys(V_pr.y, V_pr, wcm)/sigma, V_pr/V_c, label=lab, color=colours(ip), linewidth=2)
            handles.append(l1)
    
# Add axes limits and labels
ax.set_xlim(xlims)

ax.set_xlabel('$y^{*,2}/\sigma_{y,2}$')
ax.set_ylabel('$V/V_c$')

plt.legend(handles = handles, loc='upper right', title='$x/D$', frameon=False)

plt.subplots_adjust(wspace=0.18, 
                    hspace=0.25)

fig.savefig(fig_path + 'selfSimilarity/CT_onefig.pdf', bbox_inches='tight')

plt.show()

#%% Self similarity: Effect of turbulence intensity
# fraction of textwidth
fraction = 1

# Wake centre and width methods
wcm = 'max'; wwm = 'integral'

# Downstream positions
x_Ds = [3,4,5,6,7,8]

# Axes limits
xlims=(-4,4)
ylims=(None,1.1)

# Create markers
markers = ["o", "s", "D", "P", "X", "^", "v", "<", ">"]

# Generate 2 colors from the 'tab10' colormap
colours = cm.get_cmap('jet', len(x_Ds))

# Create subplot labels
subplot_labels = ['a)', 'b)', 'c)', 'd)']

# Create subplots object
fig, axs = plt.subplots(2,2,
                        figsize=set_size(textwidth, fraction), sharex=True, sharey=True)

# Make axes indexing 1D
axs = np.ravel(axs)

for iy, fc in enumerate(reversed(fc_TIs)):
    for ip, x_D in enumerate(x_Ds):
        # Extract profile
        V_pr = fc.velocityProfile('V', x_D)
        
        # Calculate wake centre velocity
        if fc.wf.yaws[0] > 0:
            _, V_c = pu.wakeCentre(wcm, V_pr.y, V_pr)
        else:
            V_c = np.max(np.abs(V_pr.to_numpy()))
            
        # Calculate wake width
        sigma = pu.wakeWidth(wwm, V_pr.y, V_pr)
        
        # Plot profile with label
        lab = '${:d}$'.format(x_D)
        if fc.wf.yaws[0] == 0:
            # axs[iy].scatter(V_pr.y/sigma, V_pr/V_c, label=lab, color=colours(ip), marker=markers[ip])
            axs[iy].plot(V_pr.y/sigma, V_pr/V_c, label=lab, color=colours(ip), linewidth=2)
        else:
            # axs[iy].scatter(pu.y_to_ys(V_pr.y, V_pr, wcm)/sigma, V_pr/V_c, label=lab, color=colours(ip), marker=markers[ip])
            axs[iy].plot(pu.y_to_ys(V_pr.y, V_pr, wcm)/sigma, V_pr/V_c, label=lab, color=colours(ip), linewidth=2)
    
    # Add axes limits and labels
    axs[iy].set_xlim(xlims)
    axs[iy].set_ylim(ylims)
    
    if iy > 1:
        axs[iy].set_xlabel('$y^{*,2}/\sigma_{y,2}$')
    if iy%2 == 0:
        axs[iy].set_ylabel('$V/V_c$')
        
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    axs[iy].text(0.12, 0.94, 
                 subplot_labels[iy] + ' $I_{{\infty,h}} = {:.2f}$'.format(fc.wf.ti),
                 transform=axs[iy].transAxes + trans,
            fontsize='medium', va='bottom',  style='italic')
    
plt.legend(loc='center left', bbox_to_anchor=(1.025,1.1), title='$x/D$', frameon=False)

plt.subplots_adjust(wspace=0.18, 
                    hspace=0.25)

fig.savefig(fig_path + 'selfSimilarity/TI.pdf', bbox_inches='tight')

plt.show()

#%% Self-similarity: effect of turbulence intensity: one figure
# fraction of textwidth
fraction = 1

# Wake centre and width methods
wcm = 'max'; wwm = 'integral'

# Downstream positions
x_Ds = [3,4,5,6,7,8]

# Axes limits
xlims=(-4,4)

# Create markers
markers = ["o", "s", "D", "P", "X", "^", "v", "<", ">"]

# Generate 2 colors from the 'tab10' colormap
colours = cm.get_cmap('jet', len(x_Ds))

# Create subplot labels
subplot_labels = ['a)', 'b)', 'c)', 'd)']

# Create subplots object
fig, ax = plt.subplots(1,1,
                        figsize=set_size(textwidth, fraction))

for iy, fc in enumerate(reversed(fc_TIs)):
    handles = []
    for ip, x_D in enumerate(x_Ds):
        # Extract profile
        V_pr = fc.velocityProfile('V', x_D)
        
        # Calculate wake centre velocity
        if fc.wf.yaws[0] > 0:
            _, V_c = pu.wakeCentre(wcm, V_pr.y, V_pr)
        else:
            V_c = np.max(np.abs(V_pr.to_numpy()))
            
        # Calculate wake width
        sigma = pu.wakeWidth(wwm, V_pr.y, V_pr)
        
        # Plot profile with label
        lab = '${:d}$'.format(x_D)
        if fc.wf.yaws[0] == 0:
            # ax.scatter(V_pr.y/sigma, V_pr/V_c, label=lab, color=colours(ip), marker=markers[ip])
            ax.plot(V_pr.y/sigma, V_pr/V_c, label=lab, color=colours(ip), linewidth=2)
        else:
            # l1 = ax.scatter(pu.y_to_ys(V_pr.y, V_pr, wcm)/sigma, V_pr/V_c, label=lab, color=colours(ip), marker=markers[ip])
            l1, = ax.plot(pu.y_to_ys(V_pr.y, V_pr, wcm)/sigma, V_pr/V_c, label=lab, color=colours(ip), linewidth=2)
            handles.append(l1)
    
# Add axes limits and labels
ax.set_xlim(xlims)

ax.set_xlabel('$y^{*,2}/\sigma_{y,2}$')
ax.set_ylabel('$V/V_c$')

plt.legend(handles = handles, loc='upper right', title='$x/D$', frameon=False)

plt.subplots_adjust(wspace=0.18, 
                    hspace=0.25)

fig.savefig(fig_path + 'selfSimilarity/TI_onefig.pdf', bbox_inches='tight')

plt.show()

#%% Distribution fitting: Effect of yaw angle: Streamwise
# fraction of textwidth
fraction = 1

# Wake centre and width methods
wcm = 'max'; wwm = 'Gaussian'

# Downstream positions
x_Ds = [3,4,5,6,7,8]

# Axes limits
xlims=(-4,4)

# Create markers
markers = ["o", "s", "D", "P", "X", "^", "v", "<", ">"]

# Generate 2 colors from the 'tab10' colormap
colours = cm.get_cmap('jet', len(x_Ds))

# Create subplot labels
subplot_labels = ['a)', 'b)', 'c)', 'd)']

# Create subplots object
fig, axs = plt.subplots(2,2,
                        figsize=set_size(textwidth, fraction), sharex=True)

# Make axes indexing 1D
axs = np.ravel(axs)

sigmas = np.empty((4,len(x_Ds)))
for iy, fc in enumerate(reversed(fc_yaws[1:5])):
    
    for ip, x_D in enumerate(x_Ds):
        # Extract profile
        U_pr = fc.velocityProfile('Udef', x_D)
        
        # Calculate wake centre velocity
        _, U_c = pu.wakeCentre(wcm, U_pr.y, U_pr)
            
        # Calculate wake width
        sigmas[iy,ip] = pu.wakeWidth(wwm, U_pr.y, U_pr)
        
        # Plot profile with label
        lab = '$x/D = {:d}$'.format(x_D)
        axs[iy].scatter(pu.y_to_ys(U_pr.y, U_pr, wcm)/sigmas[iy,ip], U_pr/U_c, label=lab, color=colours(ip), marker=markers[ip])
            
    # Fit Gaussian    
    amp = 1
    mu  = 0
    sig = np.mean(sigmas[iy,:])
    
    # Plot average fitted Gaussian
    axs[iy].plot(U_pr.y/sig, pu.Gaussian(U_pr.y, 1, 0, sig), c='k', ls='-', linewidth=2, label='Gaussian')
    
    # Add axes limits and labels
    axs[iy].set_xlim(xlims)
    
    if iy > 1:
        axs[iy].set_xlabel('$y^{*,1}/\sigma_{y,1}$')
    if iy%2 == 0:
        axs[iy].set_ylabel('$(U_{\infty}-U)/U_c$')
        
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    axs[iy].text(0.12, 0.95, 
                 subplot_labels[iy] + ' $\gamma = {:d}^\circ$'.format(fc.wf.yaws[0]),
                 transform=axs[iy].transAxes + trans,
            fontsize='medium', va='bottom', style='italic')
    
plt.legend(loc='center left', bbox_to_anchor=(1.025,1.1), frameon=False)

plt.subplots_adjust(wspace=0.18, 
                    hspace=0.25)

fig.savefig(fig_path + 'distributionFitting/yaw_U.pdf', bbox_inches='tight')

plt.show()

#%% Distribution fitting: Effect of yaw angle: Lateral
# fraction of textwidth
fraction = 1

# Wake centre and width methods
wcm = 'max'; wwm = 'integral'

# Downstream positions
x_Ds = [3,4,5,6,7,8]

# Axes limits
xlims=(-4,4)

# Create markers
markers = ["o", "s", "D", "P", "X", "^", "v", "<", ">"]

# Generate 2 colors from the 'tab10' colormap
colours = cm.get_cmap('jet', len(x_Ds))

# Create subplot labels
subplot_labels = ['a)', 'b)', 'c)', 'd)']

# Create subplots object
fig, axs = plt.subplots(2,2,
                        figsize=set_size(textwidth, fraction), sharex=True)

# Make axes indexing 1D
axs = np.ravel(axs)

sigmas = np.empty((4,len(x_Ds)))
for iy, fc in enumerate(reversed(fc_yaws[1:5])):
    
    for ip, x_D in enumerate(x_Ds):
        # Extract profile
        V_pr = fc.velocityProfile('V', x_D)
        
        # Calculate wake centre velocity
        if fc.wf.yaws[0] > 0:
            _, V_c = pu.wakeCentre(wcm, V_pr.y, V_pr)
        else:
            V_c = np.max(np.abs(V_pr.to_numpy()))
            
        # Calculate wake width
        sigmas[iy,ip] = pu.wakeWidth(wwm, V_pr.y, V_pr)
        
        # Plot profile with label
        lab = '$x/D = {:d}$'.format(x_D)
        if fc.wf.yaws[0] == 0:
            axs[iy].scatter(V_pr.y/sigmas[iy,ip], V_pr/V_c, label=lab, color=colours(ip), marker=markers[ip])
        else:
            axs[iy].scatter(pu.y_to_ys(V_pr.y, V_pr, wcm)/sigmas[iy,ip], V_pr/V_c, label=lab, color=colours(ip), marker=markers[ip])
            
    # Fit Gaussian    
    amp = 1
    mu  = 0
    sig = np.mean(sigmas[iy,:])
    
    # Plot average fitted Gaussian
    axs[iy].plot(V_pr.y/sig, pu.Gaussian(V_pr.y, 1, 0, sig), c='k', ls='-', linewidth=2, label='Gaussian')
    
    # # Plot vortex model
    # V_vm = 8 * (8/np.pi**2) * (1-np.sqrt(1-fc.wf.CTs[0])) * np.sin(np.deg2rad(-fc.wf.yaws[0])) * (1/(1+((8*V_pr.y)/(np.pi*fc.wf.D))))
    # axs[iy].plot(V_pr.y/sig, V_vm/V_c, c='k', ls='-.', label='Vortex model')
    
    # Add axes limits and labels
    axs[iy].set_xlim(xlims)
    
    if iy > 1:
        axs[iy].set_xlabel('$y^{*,2}/\sigma_{y,2}$')
    if iy%2 == 0:
        axs[iy].set_ylabel('$V/V_c$')
        
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    axs[iy].text(0.12, 0.95, 
                 subplot_labels[iy] + ' $\gamma = {:d}^\circ$'.format(fc.wf.yaws[0]),
                 transform=axs[iy].transAxes + trans,
            fontsize='medium', va='bottom', style='italic')
    
plt.legend(loc='center left', bbox_to_anchor=(1.025,1.1), frameon=False)

plt.subplots_adjust(wspace=0.18, 
                    hspace=0.25)

fig.savefig(fig_path + 'distributionFitting/yaw_V.pdf', bbox_inches='tight')

plt.show()

#%% General solution - self-similarity
# fraction of textwidth
fraction = 1

# Wake centre and width methods
wcm = 'max'; wwm = 'integral'

# Downstream positions
x_Ds = [3,4,5,6,7,8]

# Axes limits
xlims=(-4.5,4.5)

# Create markers
markers = ["o", "s", "D", "P", "X", "^", "v", "<", ">"]

# Generate 2 colors from the 'tab10' colormap
colours = cm.get_cmap('jet', len(x_Ds))

# Create subplot labels
subplot_labels = ['a)', 'b)', 'c)']

# Create subplots object
fig, axs = plt.subplots(1,3,
                        figsize=set_size(textwidth, fraction), sharey=True)

sigmas = np.empty((4,len(x_Ds)))
for iy, fc in enumerate(fc_SS):
    amps = np.empty((len(x_Ds)))
    mus  = np.empty((len(x_Ds)))
    sigs = np.empty((len(x_Ds)))
    
    for ip, x_D in enumerate(x_Ds):
        # Extract profile
        if iy == 0:
            V_pr = fc.velocityProfile('V', x_D, WT=-1)
        else:
            V_pr = fc.velocityProfile('Vdef', x_D, WT=-1)
        
        # Calculate wake centre velocity
        _, V_c = pu.wakeCentre(wcm, V_pr.y, V_pr)
            
        # Calculate wake width
        sigmas[iy,ip] = pu.wakeWidth(wwm, V_pr.y, V_pr)
        
        # Plot profile with label
        lab = '$x-x_n = {:d}D$'.format(x_D)
        if fc.wf.yaws[0] == 0:
            axs[iy].scatter(V_pr.y/sigmas[iy,ip], V_pr/V_c, label=lab, color=colours(ip), marker=markers[ip])
        else:
            axs[iy].scatter(pu.y_to_ys(V_pr.y, V_pr, wcm)/sigmas[iy,ip], V_pr/V_c, label=lab, color=colours(ip), marker=markers[ip])
        
    # Fit Gaussian
    sig = np.mean(sigmas[iy,:])
    
    # Plot average fitted Gaussian
    axs[iy].plot(V_pr.y/sig, pu.Gaussian(V_pr.y, 1, 0, sig), c='k', ls='-', linewidth=2, label='Gaussian')
    
    # Add axes limits and labels
    axs[iy].set_xlim(xlims)
    axs[iy].set_xlabel('$(y-y^{*,2})/\sigma_{y,2}$')
    axs[iy].set_xticks([-4,-2,0,2,4])
        
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    axs[iy].text(0.18, 0.98, 
                 subplot_labels[iy],
                 transform=axs[iy].transAxes + trans,
            fontsize='medium', va='bottom', style='italic')

axs[0].set_ylabel('$(V_{n} - V_{n-1})/V_c$')
    
plt.legend(loc='center left', bbox_to_anchor=(1.025,0.5), frameon=False)

plt.subplots_adjust(wspace=0.18)

fig.savefig(fig_path_derivation + 'selfSimilarity.pdf', bbox_inches='tight')

plt.show()

#%% Wake width: Effect of yaw angle
# fraction of textwidth
fraction = 1

# Wake width methods
wwm = 'integral'

# Downstream positions
x_Ds = np.linspace(5,25,n_x_D+1)

# Axes limits
xlims=(5,25)
ylims=(0,3.5)

# Create markers
markers = ["o", "s", "D", "P", "X", "^", "v", "<", ">"]

# Generate 2 colors from the 'tab10' colormap
colours = cm.get_cmap('jet', 9)

# Create subplots object
fig, ax = plt.subplots(1,1,
                        figsize=set_size(textwidth,
                                         fraction))

sigmas_u = np.empty((5,len(x_Ds)))
sigmas_v = np.empty((5,len(x_Ds)))
lines = []
for iy, fc in enumerate(reversed(fc_yaws[0:5])):
    for ip, x_D in enumerate(x_Ds):
        # Extract profile
        U_pr = fc.velocityProfile('Udef', x_D)
        V_pr = fc.velocityProfile('V', x_D)
        
        # Calculate wake widths
        sigmas_u[iy,ip] = pu.wakeWidth(wwm, U_pr.y, U_pr)
        sigmas_v[iy,ip] = pu.wakeWidth(wwm, V_pr.y, V_pr)
        
    l1, = ax.plot(x_Ds, sigmas_v[iy,:]/fc.wf.D, label='${:d}^\circ$'.format(fc.wf.yaws[0]), color=colours(iy), linewidth=2)
    lines.append(l1)
    ax.plot(x_Ds, sigmas_u[iy,:]/fc.wf.D, color=colours(iy), linewidth=2, ls='--')

ax.set_xlabel('$x/D$')
ax.set_ylabel('$\sigma_{y}/D$')
ax.set_xlim(xlims)
ax.set_ylim(ylims)

# Create legend entries for line styles
solid_line = Line2D([], [], 
                       color=colours(0),
                       ls='-',
                       label='V')
dashed_line = Line2D([], [], 
                       color=colours(0),
                       ls='--',
                       label='U')

# Add legend outside axes
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
yaw_legend = ax.legend(handles=lines, 
                       loc='center left',
                       title='$\gamma$',
                       frameon=False,
                       # bbox_to_anchor=(1.0625,0.3)
                       bbox_to_anchor=(1.025,0.3))
plt.gca().add_artist(yaw_legend)
ax.legend(handles=[solid_line, dashed_line],
          loc='center left',
          frameon=False,
          bbox_to_anchor=(1.025,0.75))

fig.savefig(fig_path + 'wakeWidth/yaw.pdf', bbox_inches='tight')

plt.show()

#%% Wake width: Effect of yaw angle: long domain
# fraction of textwidth
fraction = 1

# Wake width methods
wwm = 'integral'

# Downstream positions
x_Ds = np.linspace(5,50,n_x_D+1)

# Axes limits
xlims=(5,50)
ylims=(0,3.5)

# Create markers
markers = ["o", "s", "D", "P", "X", "^", "v", "<", ">"]

# Generate 2 colors from the 'tab10' colormap
colours = cm.get_cmap('jet', 9)

# Create subplots object
fig, ax = plt.subplots(1,1,
                        figsize=set_size(textwidth,
                                         fraction))

sigmas_u = np.empty((5,len(x_Ds)))
sigmas_v = np.empty((5,len(x_Ds)))
lines = []
for iy, fc in enumerate(reversed(fc_yaws_LD)):
    for ip, x_D in enumerate(x_Ds):
        # Extract profile
        U_pr = fc.velocityProfile('Udef', x_D)
        V_pr = fc.velocityProfile('V', x_D)
        
        # Calculate wake widths
        sigmas_u[iy,ip] = pu.wakeWidth(wwm, U_pr.y, U_pr)
        sigmas_v[iy,ip] = pu.wakeWidth(wwm, V_pr.y, V_pr)
        
    l1, = ax.plot(x_Ds, sigmas_v[iy,:]/fc.wf.D, label='${:d}^\circ$'.format(fc.wf.yaws[0]), color=colours(iy), linewidth=2)
    lines.append(l1)
    ax.plot(x_Ds, sigmas_u[iy,:]/fc.wf.D, color=colours(iy), linewidth=2, ls='--')

ax.set_xlabel('$x/D$')
ax.set_ylabel('$\sigma_{y}/D$')
ax.set_xlim(xlims)
ax.set_ylim(ylims)

# Create legend entries for line styles
solid_line = Line2D([], [], 
                       color=colours(0),
                       ls='-',
                       label='V')
dashed_line = Line2D([], [], 
                       color=colours(0),
                       ls='--',
                       label='U')

# Add legend outside axes
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
yaw_legend = ax.legend(handles=lines, 
                       loc='center left',
                       title='$\gamma$',
                       frameon=False,
                       # bbox_to_anchor=(1.0625,0.3)
                       bbox_to_anchor=(1.025,0.3))
plt.gca().add_artist(yaw_legend)
ax.legend(handles=[solid_line, dashed_line],
          loc='center left',
          frameon=False,
          bbox_to_anchor=(1.025,0.75))

fig.savefig(fig_path + 'wakeWidth/yaw_longDomain.pdf', bbox_inches='tight')

plt.show()

#%% Wake width: Effect of thrust coefficient
# fraction of textwidth
fraction = 1

# Wake width methods
wwm = 'integral'

# Downstream positions
x_Ds = np.linspace(5,25,n_x_D+1)

# Axes limits
xlims=(5,25)
ylims=(0,3.5)

# Create markers
markers = ["o", "s", "D", "P", "X", "^", "v", "<", ">"]

# Generate 2 colors from the 'tab10' colormap
colours = cm.get_cmap('jet', 9)

# Create subplots object
fig, ax = plt.subplots(1,1,
                        figsize=set_size(textwidth,
                                         fraction))

lines = []
sigmas_u = np.empty((len(CTs),len(x_Ds)))
sigmas_v = np.empty((len(CTs),len(x_Ds)))
for iy, fc in enumerate(reversed(fc_CTs)):
    for ip, x_D in enumerate(x_Ds):
        # Extract profile
        U_pr = fc.velocityProfile('Udef', x_D)
        V_pr = fc.velocityProfile('V', x_D)
        
        # Calculate wake widths
        sigmas_u[iy,ip] = pu.wakeWidth(wwm, U_pr.y, U_pr)
        sigmas_v[iy,ip] = pu.wakeWidth(wwm, V_pr.y, V_pr)
        
    l1, = ax.plot(x_Ds, sigmas_v[iy,:]/fc.wf.D, label='${:.1f}$'.format(fc.wf.CTs[0]), color=colours(iy), linewidth=2)
    lines.append(l1)
    ax.plot(x_Ds, sigmas_u[iy,:]/fc.wf.D, label='${:.1f}$'.format(fc.wf.CTs[0]), color=colours(iy), linewidth=2, ls='--')
    
ax.set_xlabel('$x/D$')
ax.set_ylabel('$\sigma_{y}/D$')
ax.set_xlim(xlims)
ax.set_ylim(ylims)

# Create legend entries for line styles
solid_line = Line2D([], [], 
                       color=colours(0),
                       ls='-',
                       label='V')
dashed_line = Line2D([], [], 
                       color=colours(0),
                       ls='--',
                       label='U')

# Add legend outside axes
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
yaw_legend = ax.legend(handles=lines, 
                       loc='center left',
                       title='$C_T$',
                       frameon=False,
                       bbox_to_anchor=(1.025,0.3))
plt.gca().add_artist(yaw_legend)
ax.legend(handles=[solid_line, dashed_line],
          loc='center left',
          frameon=False,
          bbox_to_anchor=(1.025,0.75))

fig.savefig(fig_path + 'wakeWidth/CT.pdf', bbox_inches='tight')

plt.show()

#%% Wake width: Effect of turbulence intensity: lateral
# fraction of textwidth
fraction = 1

# Wake width methods
wwm = 'integral'

# Downstream positions
x_Ds = np.linspace(5,25,n_x_D+1)

# Axes limits
xlims=(5,25)
ylims=(0,3.5)

# Create markers
markers = ["o", "s", "D", "P", "X", "^", "v", "<", ">"]

# Generate 2 colors from the 'tab10' colormap
colours = cm.get_cmap('jet', 9)

# Create subplot labels
subplot_labels = ['a)', 'b)', 'c)', 'd)']

# Create subplots object
fig, ax = plt.subplots(1,1,
                        figsize=set_size(textwidth,
                                         fraction))

sigmas_u = np.empty((len(TIs),len(x_Ds)))
sigmas_v = np.empty((len(TIs),len(x_Ds)))
lines = []
for iy, fc in enumerate(reversed(fc_TIs)):
    for ip, x_D in enumerate(x_Ds):
        # Extract profile
        U_pr = fc.velocityProfile('Udef', x_D)
        V_pr = fc.velocityProfile('V', x_D)
        
        # Calculate wake widths
        sigmas_u[iy,ip] = pu.wakeWidth(wwm, U_pr.y, U_pr)
        sigmas_v[iy,ip] = pu.wakeWidth(wwm, V_pr.y, V_pr)
        
    l1, = ax.plot(x_Ds, sigmas_v[iy,:]/fc.wf.D, label='${:.2f}$'.format(fc.wf.ti), color=colours(iy), linewidth=2)
    lines.append(l1)
    ax.plot(x_Ds, sigmas_u[iy,:]/fc.wf.D, label='${:.2f}$'.format(fc.wf.ti), color=colours(iy), linewidth=2, ls='--')

ax.set_xlabel('$x/D$')
ax.set_ylabel('$\sigma_{y}/D$')
ax.set_xlim(xlims)
ax.set_ylim(ylims)

# Create legend entries for line styles
solid_line = Line2D([], [], 
                    color=colours(0),
                    ls='-',
                    label='V')
dashed_line = Line2D([], [], 
                     color=colours(0),
                     ls='--',
                     label='U')

# Add legend outside axes
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
yaw_legend = ax.legend(handles=lines, 
                       loc='center left',
                       title='$I_{\infty,h}$',
                       frameon=False,
                       bbox_to_anchor=(1.025,0.3))
plt.gca().add_artist(yaw_legend)
ax.legend(handles=[solid_line, dashed_line],
          loc='center left',
          frameon=False,
          bbox_to_anchor=(1.025,0.75))

fig.savefig(fig_path + 'wakeWidth/TI.pdf', bbox_inches='tight')

plt.show()

#%% Wake width: Effect of turbulence intensity: lateral: normalised
# fraction of textwidth
fraction = 1

# Wake width methods
wwm = 'integral'

# Downstream positions
x_Ds = np.linspace(3,25,50)

# Axes limits
xlims=(3,25)
ylims=(0.5,3.5)

# Create markers
markers = ["o", "s", "D", "P", "X", "^", "v", "<", ">"]

# Generate 2 colors from the 'tab10' colormap
colours = cm.get_cmap('jet', 9)

# Create subplot labels
# subplot_labels = ['a)', 'b)', 'c)', 'd)']

# Create subplots object
fig, ax = plt.subplots(1,1,
                        figsize=set_size(textwidth, fraction))

# Make axes indexing 1D
# axs = np.ravel(axs)

# sigmas_u = np.empty((len(TIs),len(x_Ds)))
sigmas_v = np.empty((len(TIs),len(x_Ds)))
for iy, fc in enumerate(reversed(fc_TIs)):
    for ip, x_D in enumerate(x_Ds):
        # Extract profile
        # U_pr = fc.velocityProfile('Udef', x_D)
        V_pr = fc.velocityProfile('V', x_D)
        
        # Calculate wake widths
        # sigmas_u[iy,ip] = pu.wakeWidth(wwm, U_pr.y, U_pr)
        sigmas_v[iy,ip] = pu.wakeWidth(wwm, V_pr.y, V_pr)
        
    # ax.scatter(x_Ds, sigmas_v[iy,:]/fc.wf.D, label='${:.2f}$'.format(fc.wf.ti), edgecolors=colours(iy), facecolors=colours(iy), marker=markers[iy])
    ax.plot(x_Ds, sigmas_v[iy,:]/(fc.wf.D/np.log(fc.wf.ti)), label='${:.2f}$'.format(fc.wf.ti), color=colours(iy), linewidth=2)
    # ax.scatter(x_Ds, sigmas_u[iy,:]/fc.wf.D, label='${:.2f}$'.format(fc.wf.ti), edgecolors=colours(iy), facecolors='none', marker=markers[iy])

ax.set_xlabel('$x/D$')
ax.set_ylabel('$\sigma_{y}\ln(I_{\infty,h})/D$')
ax.set_xlim(xlims)
# ax.set_ylim(ylims)
plt.legend(title = '$I_{\infty,h}$', frameon=False)

# plt.subplots_adjust(wspace=0.18, 
                    # hspace=0.25)

fig.savefig(fig_path + 'wakeWidth/TI_V_norm.pdf', bbox_inches='tight')

plt.show()

#%% Wake centre: effect of yaw angle
# fraction of textwidth
fraction = 1

# Wake centre methods
wcm = 'Gaussian'

# Downstream positions
x_Ds = np.linspace(5,25,n_x_D-1)

# Axes limits
xlims=(5,25)

# Generate 2 colors from the 'tab10' colormap
colours = cm.get_cmap('jet', 9)

# Create subplots object
fig, ax = plt.subplots(1,1,
                        figsize=set_size(textwidth, fraction))

centres_u = np.empty((5,len(x_Ds)))
centres_v = np.empty((5,len(x_Ds)))
lines = []
for iy, fc in enumerate(reversed(fc_yaws[1:5])):
    for ip, x_D in enumerate(x_Ds):
        # Extract profile
        U_pr = fc.velocityProfile('Udef', x_D)
        V_pr = fc.velocityProfile('V', x_D)
        
        # Calculate wake centres
        centres_u[iy,ip], _ = pu.wakeCentre(wcm, U_pr.y, U_pr)
        if fc.wf.yaws[0] == 0:
            centres_v[iy,ip], _ = pu.wakeCentre('unyawed', V_pr.y, V_pr)
        else:
            centres_v[iy,ip], _ = pu.wakeCentre(wcm, V_pr.y, V_pr)
        
    l1, = ax.plot(x_Ds, centres_v[iy,:]/fc.wf.D, label='${:d}^\circ$'.format(fc.wf.yaws[0]), color=colours(iy+1), linewidth=2)
    lines.append(l1)
    ax.plot(x_Ds, centres_u[iy,:]/fc.wf.D, label='$\gamma = {:d}^\circ$'.format(fc.wf.yaws[0]), color=colours(iy+1), linewidth=2, ls='--')

ax.set_xlabel('$x/D$')
ax.set_ylabel('$(y^*-y)/D$')
ax.set_xlim(xlims)

# Create legend entries for line styles
solid_line = Line2D([], [], 
                    color=colours(0),
                    ls='-',
                    label='V')
dashed_line = Line2D([], [], 
                     color=colours(0),
                     ls='--',
                     label='U')

# Add legend outside axes
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
yaw_legend = ax.legend(handles=lines, 
                       loc='center left',
                       title='$\gamma$',
                       frameon=False,
                       bbox_to_anchor=(1.025,0.3))
plt.gca().add_artist(yaw_legend)
ax.legend(handles=[solid_line, dashed_line],
          loc='center left',
          frameon=False,
          bbox_to_anchor=(1.025,0.75))

fig.savefig(fig_path + 'wakeCentre/yaw.pdf', bbox_inches='tight')

plt.show()

#%% Wake centre: effect of thrust coefficient
# fraction of textwidth
fraction = 1

# Wake centre methods
wcm = 'Gaussian'

# Downstream positions
x_Ds = np.linspace(5,25,n_x_D+1)

# Axes limits
xlims=(5,25)

# Create markers
markers = ["o", "s", "D", "P", "X", "^", "v", "<", ">"]

# Generate 2 colors from the 'tab10' colormap
colours = cm.get_cmap('jet', 9)

# Create subplots object
fig, ax = plt.subplots(1,1,
                        figsize=set_size(textwidth, fraction))

centres_u = np.empty((len(CTs),len(x_Ds)))
centres_v = np.empty((len(CTs),len(x_Ds)))
lines = []
for iy, fc in enumerate(reversed(fc_CTs)):
    for ip, x_D in enumerate(x_Ds):
        # Extract profile
        U_pr = fc.velocityProfile('Udef', x_D)
        V_pr = fc.velocityProfile('V', x_D)
        
        # Calculate wake centres
        centres_u[iy,ip], _ = pu.wakeCentre(wcm, U_pr.y, U_pr)
        centres_v[iy,ip], _ = pu.wakeCentre(wcm, V_pr.y, V_pr)
        
    l1, = ax.plot(x_Ds, centres_v[iy,:]/fc.wf.D, label='${:.1f}$'.format(fc.wf.CTs[0]), color=colours(iy), linewidth=2)
    lines.append(l1)
    ax.plot(x_Ds, centres_u[iy,:]/fc.wf.D, label='${:.1f}$'.format(fc.wf.CTs[0]), color=colours(iy), linewidth=2, ls='--')

ax.set_xlabel('$x/D$')
ax.set_ylabel('$(y^*-y)/D$')
ax.set_xlim(xlims)

# Create legend entries for line styles
solid_line = Line2D([], [], 
                    color=colours(0),
                    ls='-',
                    label='V')
dashed_line = Line2D([], [], 
                     color=colours(0),
                     ls='--',
                     label='U')

# Add legend outside axes
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
yaw_legend = ax.legend(handles=lines, 
                       loc='center left',
                       title='$C_T$',
                       frameon=False,
                       bbox_to_anchor=(1.025,0.3))
plt.gca().add_artist(yaw_legend)
ax.legend(handles=[solid_line, dashed_line],
          loc='center left',
          frameon=False,
          bbox_to_anchor=(1.025,0.75))

fig.savefig(fig_path + 'wakeCentre/CT.pdf', bbox_inches='tight')

plt.show()

#%% Wake centre: effect of turbulence intensity
# fraction of textwidth
fraction = 1

# Wake centre methods
wcm = 'Gaussian'

# Downstream positions
x_Ds = np.linspace(5,25,n_x_D+1)

# Axes limits
xlims=(5,25)

# Create markers
markers = ["o", "s", "D", "P", "X", "^", "v", "<", ">"]

# Generate 2 colors from the 'tab10' colormap
colours = cm.get_cmap('jet', 9)

# Create subplots object
fig, ax = plt.subplots(1,1,
                        figsize=set_size(textwidth, fraction))

centres_u = np.empty((len(TIs),len(x_Ds)))
centres_v = np.empty((len(TIs),len(x_Ds)))
lines = []
for iy, fc in enumerate(reversed(fc_TIs)):
    for ip, x_D in enumerate(x_Ds):
        # Extract profile
        U_pr = fc.velocityProfile('Udef', x_D)
        V_pr = fc.velocityProfile('V', x_D)
        
        # Calculate wake centres
        centres_u[iy,ip], _ = pu.wakeCentre(wcm, U_pr.y, U_pr)
        centres_v[iy,ip], _ = pu.wakeCentre(wcm, V_pr.y, V_pr)
        
    # ax.scatter(x_Ds, centres_v[iy,:]/fc.wf.D, label='${:.2f}$'.format(fc.wf.ti), edgecolors=colours(iy), facecolors=colours(iy), marker=markers[iy])
    l1, = ax.plot(x_Ds, centres_v[iy,:]/fc.wf.D, label='${:.2f}$'.format(fc.wf.ti), color=colours(iy), linewidth=2)
    lines.append(l1)
    ax.plot(x_Ds, centres_u[iy,:]/fc.wf.D, label='${:.2f}$'.format(fc.wf.ti), color=colours(iy), linewidth=2, ls='--')

ax.set_xlabel('$x/D$')
ax.set_ylabel('$(y^*-y)/D$')
ax.set_xlim(xlims)

# Create legend entries for line styles
solid_line = Line2D([], [], 
                    color=colours(0),
                    ls='-',
                    label='V')
dashed_line = Line2D([], [], 
                     color=colours(0),
                     ls='--',
                     label='U')

# Add legend outside axes
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
yaw_legend = ax.legend(handles=lines, 
                       loc='center left',
                       title='$I_{\infty,h}$',
                       frameon=False,
                       bbox_to_anchor=(1.025,0.3))
plt.gca().add_artist(yaw_legend)
ax.legend(handles=[solid_line, dashed_line],
          loc='center left',
          frameon=False,
          bbox_to_anchor=(1.025,0.75))

fig.savefig(fig_path + 'wakeCentre/TI.pdf', bbox_inches='tight')

plt.show()