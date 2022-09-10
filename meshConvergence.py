# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 11:07:17 2022

@author: nilsg
"""
#%% Module imports
import numpy as np
import matplotlib.pyplot as plt
from post_utils import windTurbine, windFarm, flowcase, set_size, y_to_ys
from copy import copy
import xarray as xr
import matplotlib.transforms as mtransforms
from pylab import cm
from MOA import MOA

#%% Setup
# Reset to defaults
plt.rcParams.update(plt.rcParamsDefault)

# Edit the font, font size, and axes width
plt.rcParams.update({
  "text.usetex": True,
  "font.family": "Tex Gyre Adventor"
})
plt.rcParams['font.size'] = 10
plt.rcParams['lines.linewidth'] = 1.5

# Set the path for importing flowdata
res_lim = '1e-7'
data_path = 'C:/Users/nilsg/OneDrive/Documents/EWEM/Thesis/NUMERICAL/PyWakeEllipSys/flowdata/meshConvergence/' + res_lim + '/'

# Set the path for saving figures to
fig_path = 'C:/Users/nilsg/Dropbox/Apps/Overleaf/Thesis/figures/02_cfdSetup/meshConvergence/'

# Set LaTeX textwidth
textwidth = 448.0 # [pts]

#%% Disc velocity function
def vel_disc(flowdata, x_t, y_t, z_t, yaw, D, n=128):
    '''
    Function to calculate disc velocity of a turbine at (x_t, y_t, z_t) with yaw angle 'yaw' and diameter 'D' from velocity fields in 'flowdata'.

    Parameters
    ----------
    flowdata : xarray.DataSet
        xarray dataset containing velocities and coordinates.
    x_t : float
        x-coordinate of turbine rotor centre.
    y_t : float
        y-coordinate of turbine rotor centre.
    z_t : float
        z-coordinate of turbine rotor centre.
    yaw : float
        Yaw angle of rotor in radians.
    D : float
        Rotor diameter [m].
    n : int, optional
        Spatial discretisation. The default is 128.

    Returns
    -------
    vel_AD : float
        Disc velocity [m/s].

    '''
    # Find edges of rotor in x, y, z
    lower = (x_t - (D/2)*np.sin(yaw),
             y_t - (D/2)*np.cos(yaw),
             z_t - (D/2)
             )
    upper = (x_t + (D/2)*np.sin(yaw),
             y_t + (D/2)*np.cos(yaw), 
             z_t + (D/2)
             )

    # Rotated rotor coordinate axis
    r_rotor = np.linspace(y_t - D/2, y_t + D/2, n)

    # Discretise rotor surface as rectangle
    x_rotor = xr.DataArray(np.linspace(lower[0], upper[0], n),
                               dims="r", coords={"r": r_rotor})
    y_rotor = xr.DataArray(np.linspace(lower[1], upper[1], n),
                               dims="r", coords={"r": r_rotor})
    z_rotor = xr.DataArray(np.linspace(lower[2], upper[2], n),
                               dims="z")

    # 3D interpolation of rectangle around rotor
    if yaw == 0:
        flowdata_rec = flowdata.interp(x = x_t,
                                       y = y_rotor,
                                       z = z_rotor)
    else:
        flowdata_rec = flowdata.interp(x = x_rotor,
                                       y = y_rotor,
                                       z = z_rotor)

    # Filter rotor disc from rectangle
    flowdata_rotor = flowdata_rec.where(np.sqrt((flowdata_rec["r"] - y_t)**2 + (flowdata_rec.z - z_t)**2) <= (D/2),
                             other=np.nan)
    
    # Create disc averaged velocity vector (assuming W=0)
    U_AD = flowdata_rotor.U.mean()
    V_AD = abs(flowdata_rotor.V).mean()
    # V_AD = flowdata_rotor.V.mean()
    
    return U_AD, V_AD

#%% Instatiate classes
# Wind turbine
NREL5MW = windTurbine(D=126.0, zh=90.0, TSR=7.5)

# Wind farm
wf_template = windFarm(Uinf=8,
              ti=0.04,
              x_wt=[0],
              y_wt=[0],
              wts=[NREL5MW],
              yaws=[0],
              CTs=[0.8],
              wrs=[True])

#%% Settings
# Yaw angles studied
yaws = [0,25]

# Resolutions studied (per yaw angle)
cDs = [[4,8,16],
       [4,8,16]]

# Create discretisation for downstream plotting
x_Ds = np.linspace(0, 25, 101)

# Set subplot labels
subplot_labels = ['a)','b)']

#%% Preallocation
# Flowcases
fcs = [[],[]]

# Disc velocities
U_AD = np.zeros((len(yaws),
                 max([len(cDs[i]) for i in range(len(cDs))]),
                 len(x_Ds)))
V_AD = np.zeros((len(yaws),
                 max([len(cDs[i]) for i in range(len(cDs))]),
                 len(x_Ds)))

for iy, yaw in enumerate(yaws):
    #%% Import flowcases
    print('Importing flowcases for yaw = {:d}...'.format(yaw))
    for icD, cD in enumerate(cDs[iy]):
        fcs[iy].append(flowcase(dir_path=data_path + str(yaw) + '/' + str(cD) + 'cD/',
                  wf=copy(wf_template),
                  cD=cD))
        
        #%% Calculate disc velocities
        print('Calculating disc velocities for D/{:d}...'.format(cD))
        for ix, x_D in enumerate(x_Ds):
            U_AD[iy,icD,ix], V_AD[iy,icD,ix] = vel_disc(
                fcs[iy][icD].flowdata,
                (fcs[iy][icD].wf.x_wt[0]+x_D)*fcs[iy][icD].wf.D,
                fcs[iy][icD].wf.y_wt[0]*fcs[iy][icD].wf.D,
                fcs[iy][icD].wf.z_wt[0],
                fcs[iy][icD].wf.yaws[0],
                fcs[iy][icD].wf.D,
                n=256)

#%% Multi-order analysis
n_cD = max([len(cDs[i]) for i in range(len(cDs))])
# Preallocation
RE  = np.zeros((len(yaws), len(x_Ds)))
err = np.zeros((len(yaws), 
                n_cD,
                len(x_Ds)))

# Perform MOA
for iy, yaw in enumerate(yaws):
    RE[iy,:], err[iy,:len(cDs[iy]),:] = MOA(x_Ds,
                                U_AD[iy,:len(cDs[iy]),:].T,
                                cDs[iy],
                                figfile = 'MOA/yaw{:d}_U/'.format(yaw))
                                # plot_x_Ds = x_Ds[::10])

# Find index of max error
idxs_U = [np.argmax(abs(err[0,1,:]/RE[0,:])),
          np.argmax(abs(err[1,1,:]/RE[1,:]))]
    
#%% Plot MOA and errors
subplot_labels = ['a)', 'b)']

for iy, yaw in enumerate(yaws):
    # Create figure and axes objects
    fig, axs = plt.subplots(1, 2,
                           figsize = set_size(
                               width=textwidth,
                               fraction=1,
                               height_adjust=0.5),
                           sharex=True)
    
    # Create colours from colourmap
    colours = cm.get_cmap('jet', 
                          n_cD*2 + 1)
    
    # Plot Richardson extrapolation
    for i_cD, cD in enumerate(cDs[iy]):
        axs[0].plot(x_Ds,
                    U_AD[iy,i_cD,:]/fcs[iy][icD].wf.Uinf,
                    color=colours(i_cD),
                    label='$D/{:d}$'.format(cD))

    axs[0].plot(x_Ds,
                RE[iy,:]/fcs[iy][icD].wf.Uinf,
                '--k',
                label='RE')
    
    # Set axes labels
    axs[0].set_ylabel('$U_{AD}/U_{\infty,h}$')
    axs[0].set_xlabel('$x/D$')
    
    # Set axes limits
    axs[0].set_xlim((0,25))
    
    # Plot errors
    for i_cD, cD in enumerate(cDs[iy]):
        axs[1].plot(x_Ds,
                    (err[iy,i_cD,:]/RE[iy,:])*100,
                    color=colours(i_cD),
                    label='$D/{:d}$'.format(cD))
    
    axs[1].plot(x_Ds,
                np.zeros(len(x_Ds)),
                '--k',
                label='RE')
    
    axs[1].set_xlabel('$x/D$')
    axs[1].set_ylabel(r'$\varepsilon_{RE,n}$ [\%]')
    
    fig.subplots_adjust(wspace=0.35)
    
    axs[1].legend(loc='center left',
                  bbox_to_anchor=(1.02, 0.5),
                  frameon=False,
                  title='Cell size')
    
    # Add subplot labels
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    for i, ax in enumerate(axs):
        ax.text(0.0, 1.05, 
                     subplot_labels[i],
                     transform = ax.transAxes + trans,
                fontsize='medium', va='bottom', style='italic')
    
    fig.savefig(fig_path  + str(yaw) + '/U_AD.pdf',
                bbox_inches='tight')
    
    plt.show()

#%% Multi-order analysis (V)
n_cD = max([len(cDs[i]) for i in range(len(cDs))])
# Preallocation
RE  = np.zeros((len(yaws), len(x_Ds)))
err = np.zeros((len(yaws), 
                n_cD,
                len(x_Ds)))

# Perform MOA
for iy, yaw in enumerate(yaws):
    RE[iy,:], err[iy,:len(cDs[iy]),:] = MOA(x_Ds,
                                V_AD[iy,:len(cDs[iy]),:].T,
                                cDs[iy],
                                figfile = 'fig/MOA/yaw{:d}_V/'.format(yaw))
                                # plot_x_Ds = x_Ds[::10])
                                
# Find index of max error
idxs_V = [np.argmax(abs(err[0,1,:]/RE[0,:])),
          np.argmax(abs(err[1,1,:]/RE[1,:]))]
    
#%% Plot MOA and errors
for iy, yaw in enumerate(yaws):
    # Create figure and axes objects
    fig, axs = plt.subplots(1, 2,
                           figsize = set_size(
                               width=textwidth,
                               fraction=1,
                               height_adjust=0.5),
                           sharex=True)
    
    # Create colours from colourmap
    colours = cm.get_cmap('jet', 
                          n_cD*2 + 1)
    
    # Plot Richardson extrapolation
    for i_cD, cD in enumerate(cDs[iy]):
        # axs[0].semilogy(x_Ds,
        #             V_AD[iy,i_cD,:]/fcs[iy][icD].wf.Uinf,
        #             color=colours(i_cD),
        #             label='$D/{:d}$'.format(cD))
        if yaw == 0:
            axs[0].semilogy(x_Ds,
                        V_AD[iy,i_cD,:]/fcs[iy][icD].wf.Uinf,
                        color=colours(i_cD),
                        label='$D/{:d}$'.format(cD))
        else:
            axs[0].plot(x_Ds,
                        V_AD[iy,i_cD,:]/fcs[iy][icD].wf.Uinf,
                        color=colours(i_cD),
                        label='$D/{:d}$'.format(cD))

    axs[0].plot(x_Ds,
                RE[iy,:]/fcs[iy][icD].wf.Uinf,
                '--k',
                label='RE')
    
    # Set axes labels
    axs[0].set_ylabel('$|V|_{AD}/U_{\infty,h}$')
    axs[0].set_xlabel('$x/D$')
    
    # Set axes limits
    axs[0].set_xlim((0,25))
    # if yaw == 0:
        # axs[0].set_ylim((-1e-7, 1e-7))
    # else:
    #     axs[0].set_ylim((0, 0.06))
    
    # Plot errors
    for i_cD, cD in enumerate(cDs[iy]):
        axs[1].plot(x_Ds,
                    (err[iy,i_cD,:]/RE[iy,:])*100,
                    color=colours(i_cD),
                    label='$D/{:d}$'.format(cD))
    
    axs[1].plot(x_Ds,
                np.zeros(len(x_Ds)),
                '--k',
                label='RE')
    
    axs[1].set_xlabel('$x/D$')
    axs[1].set_ylabel(r'$\varepsilon_{RE,n}$ [\%]')
    
    # Set axes limits
    # if yaw == 0:
        # axs[1].set_ylim((-50, 50))
    # else:
    #     axs[1].set_ylim((-10, 10))
    
    if yaw == 0:
        fig.subplots_adjust(wspace=0.35)
    else:
        fig.subplots_adjust(wspace=0.4)
        
    axs[1].legend(loc='center left',
                  bbox_to_anchor=(1.02, 0.5),
                  frameon=False,
                  title='Cell size')
    
    # Add subplot labels
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    for i, ax in enumerate(axs):
        ax.text(0.0, 1.05, 
                     subplot_labels[i],
                     transform = ax.transAxes + trans,
                fontsize='medium', va='bottom', style='italic')
    
    fig.savefig(fig_path  + str(yaw) + '/V_AD.pdf',
                bbox_inches='tight')
    
    plt.show()
    
#%% Debugging
# error = ((U_AD[0,:,:] - RE[0,:])/RE[0,:])*100
# print(error)

#%% Plot velocity profiles for different cDs
x_D_pr_U = x_Ds[idxs_U]
x_D_pr_V = x_Ds[idxs_V]

x_D_pr_U = [6, 6]
x_D_pr_V = [6, 6]

for iy, yaw in enumerate(yaws):
    # Generate colors from the 'jet' colormap
    colours = cm.get_cmap('jet', len(cDs[iy])*2+1)
    
    # Create figures and axes objects
    fig, axs = plt.subplots(1,2,figsize=set_size(width=textwidth,
                                                fraction=1,
                                                height_adjust=0.5))
        
    for icD, cD in enumerate(cDs[iy]): 
        # Extract profile
        print('Extracting velocity profiles for D/{:d}...'.format(cD))
        U_pr = fcs[iy][icD].velocityProfile('U', x_D_pr_U[iy], WT=0)
        V_pr = fcs[iy][icD].velocityProfile('V', x_D_pr_V[iy], WT=0)
        
        if yaw == 0:
            axs[0].plot(U_pr.y/fcs[iy][icD].wf.D,
                    1-(U_pr/fcs[iy][icD].wf.Uinf),
                    label='$D/'+str(cD)+'$',
                    color=colours(icD))
        else:
            axs[0].plot(y_to_ys(U_pr.y, fcs[iy][icD].wf.Uinf-U_pr, 'Gaussian')/fcs[iy][icD].wf.D,
                    1-(U_pr/fcs[iy][icD].wf.Uinf),
                    label='$D/'+str(cD)+'$',
                    color=colours(icD))
        axs[1].plot(V_pr.y/fcs[iy][icD].wf.D,
                    V_pr/fcs[iy][icD].wf.Uinf,
                    label='$D/'+str(cD)+'$',
                    color=colours(icD))
    
    # Set axes labels
    if yaw == 0:
        axs[0].set_xlabel('$y/D$')
    else:
        axs[0].set_xlabel('$y^{*,1}/D$')
    axs[1].set_xlabel('$y/D$')
    axs[0].set_ylabel('$1-(U/U_{\infty,h})$')
    axs[1].set_ylabel('$V/U_{\infty,h}$')
    
    # Set axes limits
    axs[0].set_xlim((-2,2))
    axs[1].set_xlim((-2,2))
    
    # Add subplot labels
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    for i, ax in enumerate(axs):
        ax.text(0.15, 0.96, 
                     subplot_labels[i],
                     transform = ax.transAxes + trans,
                fontsize='medium', va='bottom', style='italic')
    
    # Adjust spacing between subplots
    if yaw == 0:
        fig.subplots_adjust(wspace=0.4)
    else:
        fig.subplots_adjust(wspace=0.35)
    
    # Add single legend for both subplots outside of axes
    axs[1].legend(loc='center left', bbox_to_anchor=(1.03, 0.5), title='Cell size', frameon=False)
    
    # Save figure
    fig.savefig(fig_path + str(yaw) + '/vel_pr.pdf', 
                bbox_inches='tight')
        
    # Show figure
    plt.show()