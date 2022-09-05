# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 16:31:51 2022

@author: nilsg
"""

#%% Module imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from post_utils import windTurbine, windFarm, flowcase, set_size, wakeWidth, draw_AD
import copy
import xarray as xr
import warnings
from pylab import cm
from analytical_functions import turb_add, epsilon

# Suppress FutureWarnings (for xarray)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Edit the font, font size, and axes width
plt.rcParams.update({
  "text.usetex": True,
  "font.family": "Tex Gyre Adventor"
})
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1

data_path = 'C:/Users/nilsg/OneDrive/Documents/EWEM/Thesis/NUMERICAL/PyWakeEllipSys/flowdata/wakeWidth/'

fig_path = 'C:/Users/nilsg/Dropbox/Apps/Overleaf/Thesis/figures/05_modelImplementation/'

textwidth = 448.0 # [pts]

#%% Wind turbine & wind farm objects

NREL5MW = windTurbine(D=126.0, zh=90.0, TSR=7.5)

wf_1row = windFarm(Uinf=8,
                      ti=0.06,
                      x_wt=[ 0,  0,  0],
                      y_wt=[ 4,  -4,  0],
                      wts=[NREL5MW]*3,
                      yaws=[0]*3,
                      CTs=[0.8]*3,
                      wrs=[True]*3)

wf_2row = windFarm(Uinf=8,
                      ti=0.06,
                      x_wt=[ 0,  0,  0,
                             7,  7,  7],
                      y_wt=[ 4,  -4,  0,
                             4,  -4,  0],
                      wts=[NREL5MW]*6,
                      yaws=[0]*6,
                      CTs=[0.8]*6,
                      wrs=[True]*6)

wf_3row = windFarm(Uinf=8,
                      ti=0.06,
                      x_wt=[ 0,  0,  0,
                             7,  7,  7,
                            14, 14, 14],
                      y_wt=[ 4,  -4,  0,
                             4,  -4,  0,
                             4,  -4,  0],
                      wts=[NREL5MW]*9,
                      yaws=[0]*9,
                      CTs=[0.8]*9,
                      wrs=[True]*9)

wf_4row = windFarm(Uinf=8,
                      ti=0.06,
                      x_wt=[ 0,  0,  0,
                             7,  7,  7,
                            14, 14, 14,
                            21, 21, 21],
                      y_wt=[ 4,  -4,  0,
                             4,  -4,  0,
                             4,  -4,  0,
                             4,  -4,  0],
                      wts=[NREL5MW]*12,
                      yaws=[0]*12,
                      CTs=[0.8]*12,
                      wrs=[True]*12)

wf_5row = windFarm(Uinf=8,
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
                      yaws=[0]*15,
                      CTs=[0.8]*15,
                      wrs=[True]*15)

#%% Import flowcases
print('Importing flowcases for yaw = 0...')
yaw0_oneRow = flowcase(dir_path=data_path + '0/1/',
                  wf=copy.copy(wf_1row),
                  cD=8)

yaw0_twoRows = flowcase(dir_path=data_path + '0/2/',
                  wf=copy.copy(wf_2row),
                  cD=8)

yaw0_threeRows = flowcase(dir_path=data_path + '0/3/',
                  wf=copy.copy(wf_3row),
                  cD=8)

yaw0_fourRows = flowcase(dir_path=data_path + '0/4/',
                  wf=copy.copy(wf_4row),
                  cD=8)

yaw0_fiveRows = flowcase(dir_path=data_path + '0/5/',
                  wf=copy.copy(wf_5row),
                  cD=8)

yaw0  = [yaw0_oneRow, yaw0_twoRows, yaw0_threeRows, yaw0_fourRows, yaw0_fiveRows]

print('Importing flowcases for yaw = 25...')
yaw25_oneRow = flowcase(dir_path=data_path + '25/1/',
                  wf=copy.copy(wf_1row),
                  cD=8)
yaw25_oneRow.wf.yaws = np.deg2rad([25]*3)

yaw25_twoRows = flowcase(dir_path=data_path + '25/2/',
                  wf=copy.copy(wf_2row),
                  cD=8)
yaw25_twoRows.wf.yaws = np.deg2rad([25]*6)

yaw25_threeRows = flowcase(dir_path=data_path + '25/3/',
                  wf=copy.copy(wf_3row),
                  cD=8)
yaw25_threeRows.wf.yaws = np.deg2rad([25]*9)

yaw25_fourRows = flowcase(dir_path=data_path + '25/4/',
                  wf=copy.copy(wf_4row),
                  cD=8)
yaw25_fourRows.wf.yaws = np.deg2rad([25]*12)

yaw25_fiveRows = flowcase(dir_path=data_path + '25/5/',
                  wf=copy.copy(wf_5row),
                  cD=8)
yaw25_fiveRows.wf.yaws = np.deg2rad([25]*15)

yaw25  = [yaw25_oneRow, yaw25_twoRows, yaw25_threeRows, yaw25_fourRows, yaw25_fiveRows]

# yaw25  = [yaw25_oneRow, yaw25_twoRows, yaw25_threeRows, yaw25_fiveRows]

#%% Disc-averaged tke
def I_disc(flowdata, x_t, y_t, z_t, yaw, D, n=128):
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
    flowdata_rec = flowdata.interp(x=x_rotor, y=y_rotor, z=z_rotor)

    # Filter rotor disc from rectangle
    flowdata_rotor = flowdata_rec.where(np.sqrt((flowdata_rec["r"] - y_t)**2 + (flowdata_rec.z - z_t)**2) <= (D/2),
                             other=np.nan)
    
    # Create disc averaged quantities
    k = flowdata_rotor.tke.mean()
    U = flowdata_rotor.U.mean()
    V = flowdata_rotor.V.mean()
    W = flowdata_rotor.W.mean()
    
    I_AD = np.sqrt((2/3)*k) / np.sqrt(U**2 + V**2 + W**2)
    Iu_AD = I_AD * 0.8
    
    return Iu_AD

#%% Majid's Figure 7
I_in = np.ones((2,5))
dI_in = np.zeros((2,5))
dI_Crespo = np.zeros((2,5))
k_U = np.zeros((2,5))
k_V = np.zeros((2,5))
print('Plotting...')
for iy, fcs in enumerate([yaw0, yaw25]):
    if iy == 0:
        fig_U, axs_U = plt.subplots(2,2,figsize=set_size(width=textwidth,
                                                    fraction=1))
        gs = axs_U[0,0].get_gridspec() # get geometry of subplot grid
        
        # Remove top row of subplot
        for ax in axs_U[0,:]:
            ax.remove()
        
        # Replace with a single plot spanning the whole width
        axbig_U = fig_U.add_subplot(gs[0,:])
    else:
        fig_U, axs_U = plt.subplots(2,1,figsize=set_size(width=textwidth,
                                                    fraction=1))
    
    fig_V, axs_V = plt.subplots(2,1,figsize=set_size(width=textwidth,
                                                fraction=1))
    
    # Colours
    colours = cm.get_cmap('jet', len(fcs)*2+1)
    
    # Markers
    markers = ["o", "s", "D", "P", "X", "^", "v", "<", ">"]
    
    # Create subplot labels
    subplot_labels = ['a)', 'b)', 'c)', 'd)']
    
    # Plot the variation of wake width
    x_Ds = np.arange(3,8.5,0.5)
    sigma_U = np.zeros((len(fcs),len(x_Ds)))
    sigma_V = np.zeros((len(fcs),len(x_Ds)))
    for i_rows, fc in enumerate(fcs):
        # Calculate incoming and added turbulence intensity
        I_in[iy,i_rows] = I_disc(fc.flowdata_remove[-1],
                        fc.wf.x_wt[-1]*fc.wf.D,
                        fc.wf.y_wt[-1]*fc.wf.D,
                        fc.wf.z_wt[-1],
                        fc.wf.yaws[-1],
                        fc.wf.D)
        
        if I_in[iy,i_rows] < fc.wf.ti*0.8:
            dI_in[iy,i_rows] = 0
        else:
            dI_in[iy,i_rows] = np.sqrt(I_in[iy,i_rows]**2 - (fc.wf.ti*0.8)**2)
        
        # Calculate wake widths
        for ix, x_D in enumerate(x_Ds):
            # Extract velocity profile
            U_pr = fc.velocityProfile('Udef', x_D, -1)
            V_pr = fc.velocityProfile('V', x_D, -1)
            
            # Calculate wake width
            sigma_U[i_rows, ix] = wakeWidth(method='integral',
                                          y=U_pr.y,
                                          vel=U_pr)
            sigma_V[i_rows, ix] = wakeWidth(method='integral',
                                          y=V_pr.y,
                                          vel=V_pr)
            
        # Make linear fit of wake widths
        coef_U = np.polyfit(x_Ds, sigma_U[i_rows,:]/fc.wf.D, 1)
        k_U[iy,i_rows] = coef_U[0]
        fit_U  = np.poly1d(coef_U)
        
        coef_V = np.polyfit(x_Ds, sigma_V[i_rows,:]/fc.wf.D, 1)
        k_V[iy,i_rows] = coef_V[0]
        fit_V  = np.poly1d(coef_V)
        
        # Calculate added TI from Crespo model
        s = k_U[iy,i_rows-1] * (fc.wf.x_wt[i_rows] - fc.wf.x_wt[i_rows-1])*fc.wf.D + epsilon(0.8) * fc.wf.D
        dI_Crespo[iy,i_rows] = turb_add(fc.flowdata.y,
                             fc.flowdata.z,
                             0,
                             fc.wf.zh,
                             fc.wf.yaws[0],
                             s,
                             0.8,
                             0,
                             fc.wf.zh,
                             fc.wf.yaws[0],
                             fc.wf.D,
                             7*fc.wf.D,
                             fc.wf.ti*0.8)
        dI_Crespo[iy,0] = 0
        
        if iy == 0:
            axbig_U.scatter(x_Ds, sigma_U[i_rows,:]/fc.wf.D,
                          marker=markers[i_rows+1],
                          color=colours(i_rows),
                          label='Row {:d} ($n={:d}$)'.format(i_rows+1,3*(i_rows+1)))
            axbig_U.plot(x_Ds, fit_U(x_Ds),
                          color=colours(i_rows))
        else:
            axs_U[0].scatter(x_Ds, sigma_U[i_rows,:]/fc.wf.D,
                          marker=markers[i_rows+1],
                          color=colours(i_rows),
                          label='Row {:d} ($n={:d}$)'.format(i_rows+1,3*(i_rows+1)))
            axs_U[0].plot(x_Ds, fit_U(x_Ds),
                          color=colours(i_rows))
        
        axs_V[0].scatter(x_Ds, sigma_V[i_rows,:]/fc.wf.D,
                      marker=markers[i_rows+1],
                      color=colours(i_rows),
                      label='Row {:d} ($n={:d}$)'.format(i_rows+1,3*(i_rows+1)))
        
        axs_V[0].plot(x_Ds, fit_V(x_Ds),
                      color=colours(i_rows))
    
    # Plot k/I_in
    if iy == 0:
        axs_U[1,0].scatter(np.arange(1,6),
                         k_U[iy,:]/I_in[iy,:],
                         label='RANS',
                         clip_on = False,
                         color = 'k',
                         zorder = 10)
        
        axs_U[1,0].axhline(np.mean(k_U[iy,1:]/I_in[iy,1:]),
                           ls='--', color='k',
                           label=r'$k = {:.1f}I$'.format(np.mean(k_U[iy,1:]/I_in[iy,1:])))
        
        axs_U[1,0].legend()
    else:
        axs_U[1].scatter(np.arange(1,6),
                         k_U[iy,:]/I_in[iy,:],
                         label='RANS',
                         clip_on = False,
                         color = 'k',
                         zorder = 10)
    
    axs_V[1].scatter(np.arange(1,6),
                     k_V[iy,:]/I_in[iy,:],
                     label='RANS',
                     clip_on = False,
                     color = 'k',
                     zorder = 10)
    
    axs_V[1].axhline(np.mean(k_V[iy,1:]/I_in[iy,1:]),
                       ls='--', color='k',
                       label=r'$k = {:.1f}I$'.format(np.mean(k_V[iy,1:]/I_in[iy,1:])))
    
    axs_V[1].legend()
    
    # Plot added turbulence intensities
    if iy == 0:
        axs_U[1,1].scatter(np.arange(1,6),
                         dI_in[iy,:],
                         label='RANS',
                         color = 'k',
                         clip_on = False,
                         zorder = 10)
        
        axs_U[1,1].plot(np.arange(1,6),
                         dI_Crespo[iy,:],
                         ls = '-',
                         label='Mod. Crespo model',
                         color = 'r',
                         marker = 's',
                         clip_on = False,
                         zorder = 5)
        
        axs_U[1,1].legend()
    
        fig_U.subplots_adjust(wspace = 0.4,
                            hspace = 0.5)
    else:
        fig_U.subplots_adjust(hspace = 0.5)
    
    fig_V.subplots_adjust(wspace = 0.4,
                        hspace = 0.5)
    
    if iy == 0:
        axbig_U.set_xlabel('$(x-x_n)/D$')
        axbig_U.set_ylabel('$\sigma_{n,1}/D$')
        axbig_U.legend(frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.00))
        trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig_U.dpi_scale_trans)
        axbig_U.text(0.025, 0.95, 
                     subplot_labels[0],
                     transform=axbig_U.transAxes + trans,
                fontsize='medium', va='bottom',  style='italic')
        
        for i in range(len(axs_U[1,:])):
            trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig_U.dpi_scale_trans)
            axs_U[1,i].text(0.13, 0.95, 
                         subplot_labels[i+1],
                         transform=axs_U[1,i].transAxes + trans,
                    fontsize='medium', va='bottom',  style='italic')
            
            axs_U[1,i].set_xlabel('Row number')
            axs_U[1,i].set_xticks(np.arange(1,6))
            axs_U[1,i].set_xlim((1,5))
        axs_U[1,0].set_ylabel('$k_{w,1}/I_{in}$')
        axs_U[1,1].set_ylabel('$\Delta I_{in}$')
        
        fig_U.savefig(fig_path + 'yaw{:d}_'.format(int(np.rad2deg(fcs[0].wf.yaws[0]))) + 'wakeWidthAndTI_U' + '.pdf', bbox_inches='tight')
    else:
        axs_U[0].set_xlabel('$(x-x_n)/D$')
        axs_U[0].set_ylabel('$\sigma_{n,1}/D$')
        axs_U[0].legend(frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.00))
        trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig_U.dpi_scale_trans)
        axs_U[0].text(0.025, 0.95, 
                     subplot_labels[0],
                     transform=axs_U[0].transAxes + trans,
                fontsize='medium', va='bottom',  style='italic')
        
        trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig_U.dpi_scale_trans)
        axs_U[1].text(0.025, 0.95, 
                     subplot_labels[1],
                     transform=axs_U[1].transAxes + trans,
                fontsize='medium', va='bottom',  style='italic')
        
        axs_U[1].set_xlabel('Row number')
        axs_U[1].set_xticks(np.arange(1,6))
        axs_U[1].set_xlim((1,5))
        axs_U[1].set_ylabel('$k_{w,1}/I_{in}$')
        
        fig_U.savefig(fig_path + 'yaw{:d}_'.format(int(np.rad2deg(fcs[0].wf.yaws[0]))) + 'wakeWidthAndTI_U' + '.pdf', bbox_inches='tight')
        
    axs_V[0].set_xlabel('$(x-x_n)/D$')
    
    axs_V[0].set_ylabel('$\sigma_{n,2}/D$')
    
    axs_V[0].legend(frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.00))
    
    
    
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig_V.dpi_scale_trans)
    axs_V[0].text(0.025, 0.95, 
                 subplot_labels[0],
                 transform=axs_V[0].transAxes + trans,
            fontsize='medium', va='bottom',  style='italic')
    
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig_V.dpi_scale_trans)
    axs_V[1].text(0.025, 0.95, 
                 subplot_labels[1],
                 transform=axs_V[1].transAxes + trans,
            fontsize='medium', va='bottom',  style='italic')
    
    axs_V[1].set_xlabel('Row number')
    axs_V[1].set_xticks(np.arange(1,6))
    axs_V[1].set_xlim((1,5))
    axs_V[1].set_ylabel('$k_{w,2}/I_{in}$')
    
    fig_V.savefig(fig_path + 'yaw{:d}_'.format(int(np.rad2deg(fcs[0].wf.yaws[0]))) + 'wakeWidthAndTI_V' + '.pdf', bbox_inches='tight')

plt.show()

#%% Plot contour of turbulence intensity
for iy, fcs in enumerate([yaw0, yaw25]):
    for i_rows, fc in enumerate(fcs):
        fig, axs = plt.subplots(2,1,sharex=True,
                                figsize=set_size(width=textwidth,
                                                 fraction=1))
        axs[0].contourf(fc.flowdata.x/fc.wf.D,
                    fc.flowdata.y/fc.wf.D,
                    fc.flowdata.tke.interp(z=fc.wf.zh).T,
                    cmap='jet')
        for i_t in range(fc.wf.n_wt):
            draw_AD(axs[0], 'top', fc.wf.x_AD[i_t],
                    fc.wf.y_wt[i_t], fc.wf.D/fc.wf.D, fc.wf.yaws[0])
        
        axs[1].contourf(fc.flowdata_remove[-1].x/fc.wf.D,
                    fc.flowdata_remove[-1].y/fc.wf.D,
                    fc.flowdata_remove[-1].tke.interp(z=fc.wf.zh).T,
                    cmap='jet')
        for i_t in range(fc.wf.n_wt-1):
            draw_AD(axs[1], 'top', fc.wf.x_AD[i_t],
                    fc.wf.y_wt[i_t], fc.wf.D/fc.wf.D, fc.wf.yaws[0])
        
        axs[-1].set_xlabel('$x/D$')
        for ax in axs:
            ax.set_ylabel('$y/D$')
            ax.set_xlim((min(fc.wf.x_AD)-2, max(fc.wf.x_AD)+10))
            ax.set_ylim((min(fc.wf.y_wt)-2, max(fc.wf.y_wt)+2))
            
#%% Debugging