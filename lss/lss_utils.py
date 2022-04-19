# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 13:02:38 2022

@author: nilsg
"""

import xarray
import warnings
import numpy as np
import scipy.interpolate as sp
import scipy.integrate as spi
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors
import matplotlib.lines as mlines

mcolours = mcolors.TABLEAU_COLORS

warnings.simplefilter(action='ignore', category=FutureWarning)

class flowcase():
    def __init__(self, Uinf, D, zh, x_AD, y_AD, cD, wd=[(4,10),(4,4)]):
        '''
        

        Parameters
        ----------
        D : TYPE
            DESCRIPTION.
        zh : TYPE
            DESCRIPTION.
        x_AD : TYPE
            DESCRIPTION.
        y_AD : TYPE
            DESCRIPTION.
        wd : TYPE, optional
            DESCRIPTION. The default is [(4,10),(4,4)].

        Returns
        -------
        None.

        '''
        # Simulation parameters
        self.cD   = cD # number of cells per diameter [-]
        
        # Freestream velocity
        self.Uinf = Uinf # [m/s]
        
        # Turbine parameters
        self.D    = D  # diameter [m]
        self.zh   = zh # hub height [m]
        
        # Wind farm layout (PyWakeEllipSys convention)
        self.x_AD = np.array(x_AD) - (x_AD[-1]-x_AD[0])/2
        self.y_AD = np.array(y_AD)
        
        # Wake domain
        self.x_wd = np.asarray([self.x_AD[0]-wd[0][0]*self.D, self.x_AD[-1]+wd[0][1]*self.D])
        self.y_wd = np.array([self.y_AD[0]-wd[1][0]*self.D, self.y_AD[0]+wd[1][1]*self.D])

    def load_flowdata(self,filename):
        # Load full 3D flow data as xarray.Dataset
        self.flowdata = xarray.open_dataset(filename)
        # Define useful variables
        self.U_wd = wakeDomain(self.flowdata.U,self.x_wd,self.y_wd)
        self.U_zh = self.U_wd.interp(z=self.zh)
        self.V_wd = wakeDomain(self.flowdata.V,self.x_wd,self.y_wd)
        self.V_zh = self.V_wd.interp(z=self.zh)
            
    def plot_velProf2(self, yaw, poss, ylim=(-3,3), xlim=(-2,10), FigureSize=(6.4,4.8)):
        fig, axs = plt.subplots(2,len(poss),figsize=FigureSize,sharey=True)
        gs = axs[0,0].get_gridspec() # get geometry of subplot grid
        
        # Remove top row of subplots
        for ax in axs[0,:]:
            ax.remove()
        
        # Replace with a single plot spanning the whole width
        axbig = fig.add_subplot(gs[0,:])
        
        # Plot contour of lateral velocity at hub height
        p = axbig.contourf(self.V_zh.x/self.D, self.V_zh.y/self.D, self.V_zh.T, levels=100)
        draw_AD(axbig,0,0,self.D/self.D,yaw)
        
        # Set axes limits
        axbig.set_xlim(xlim)
        axbig.set_ylim(ylim)
        
        # Add colourbar
        divider = make_axes_locatable(axbig)
        cax     = divider.append_axes('right', size='5%', pad=0.05)
        cbar    = fig.colorbar(p, ax=axbig, cax=cax)
        cbar.set_label('$V$ [m/s]')
        
        # Preallocate for running maximum / minimum
        Vmax = -np.inf
        Vmin = 0
        
        for ip, pos in enumerate(poss):
            # Add lines to contour plot to show planes
            axbig.vlines(pos, ylim[0], ylim[1], color='k', ls='--')
            
            # Extract profile velocities
            V_pr = self.V_zh.interp(x = self.x_AD[0] + pos*self.D)
            
            # Plot profiles
            if yaw == 0:
                axs[1,ip].plot(V_pr,V_pr.y/self.D) # y
            else:
                axs[1,ip].plot(V_pr,y_to_ys(V_pr.y.to_numpy(), V_pr.values)/self.D) # y*
            
            # Add axes limits and labels
            axs[1,ip].set_ylim(ylim)
            axs[1,ip].set_xlabel('$V$ [m/s]')
            
            # Label each plot with x/D
            lab = r'$%.0fD$' % pos
            axs[1,ip].text(0.98,0.95,lab,horizontalalignment='right',verticalalignment='top',transform=axs[1,ip].transAxes)
            
            # calculate overall velocity limits
            mn = np.nanmin(V_pr.values)
            if mn < Vmin:
                Vmin = mn
            mx = np.nanmax(V_pr.values)
            if mx > Vmax:
                Vmax = mx
        
        # set velocity axis limits
        for ip in range(len(poss)):
            axs[1,ip].set_xlim(Vmin*1.05,Vmax*1.05)
        
        if yaw == 0:
            axs[1,0].set_ylabel('$y/D$ [-]')
        else:
            axs[1,0].set_ylabel('$y*/D$ [-]')
        plt.suptitle("$V_{z_h}$ for $\gamma = %d^\circ$" % yaw)
        
        # Save figure (.pdf and .svg)
        fig.tight_layout()
        filename = 'velProf2_%d' % yaw
        fig.savefig('fig/' + filename + '.pdf',bbox_inches='tight')
        fig.savefig('fig/' + filename + '.svg',bbox_inches='tight')
    
    def plot_LSS_Gaussian(self, poss, wwm, ylim=(0,None), xlim=None, FigureSize=(6.4,4.8)):
        wwm_labels = {
            'half-width': 'r_{1/2}',
            '1%'        : 'r_{1\%}',
            'Gaussian'  : '\sigma_y',
            'integral'  : '\sigma_y'
            }
        
        fig, ax = plt.subplots(1,1,figsize=FigureSize)
        
        amp   = np.empty((len(poss)))
        mu    = np.empty((len(poss)))
        sig   = np.empty((len(poss)))
        
        for ip, pos in enumerate(poss):
            # Extract profile
            V_pr = self.V_zh.interp(x = self.x_AD[0] + pos*self.D)
            
            # Calculate std
            mask  = ~np.isnan(V_pr.values)
            xdata = y_to_ys(V_pr.y.to_numpy()[mask], V_pr.values[mask])
            ydata = np.abs(V_pr.values[mask])
            sigma = wakewidth(wwm, xdata, ydata)
        
            # Plot profile with label
            lab = '$x/D = %d$' % pos
            ax.plot(y_to_ys(V_pr.y.to_numpy(),V_pr.values)/sigma,V_pr/np.nanmax(V_pr),label=lab)
            
            # Fit Gaussian
            if wwm == 'Gaussian' or 'integral':
                amp[ip], mu[ip], sig[ip] = fit_Gaussian(xdata,ydata)
                # amp[ip], mu[ip], sig[ip] = fit_Gaussian(y_to_ys(V_pr.y.to_numpy(),V_pr.values)/sigma,V_pr/np.nanmax(V_pr))
        sig = np.mean(sig)
        ax.plot(np.linspace(np.nanmin(xdata)/sig,np.nanmax(xdata)/sig,100),Gauss(np.linspace(np.nanmin(xdata)/sig,np.nanmax(xdata)/sig,100),1,0,sig/sig),c='k',ls='--',label='Gaussian \n($\sigma = %.1f$)' % (sig/sig))
        # Add axes limits and labels
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('$y*/%s$ [-]' % wwm_labels.get(wwm, 'r'))
        ax.set_ylabel('$|V|/|V_{max}|$ [-]')
        ax.set_title('$\gamma = 40^\circ$, $C_T = 1.0$, T.I.$= 0.1$')
        
        # Add legend
        ax.legend()
    
        # Save figure (.pdf and .svg)
        # fig.tight_layout()
        plt.show()
        filename = 'plot4'
        fig.savefig('fig/' + filename + '.pdf',bbox_inches='tight')
        fig.savefig('fig/' + filename + '.svg',bbox_inches='tight')
        
        return sig
    
    def plot_wakeCentresandEdges(self, x_Ds, wwm, yaw, xlim=None, ylim=None, FigureSize=(6.4,4.8)):
        y_U = []
        y_V = []
        sigma_U = []
        sigma_V = []
        
        for i, x_D in enumerate(x_Ds):
            # Extract profiles
            U_pr = self.Uinf - self.U_zh.interp(x = self.x_AD[0] + x_D*self.D)
            V_pr = self.V_zh.interp(x = self.x_AD[0] + x_D*self.D)
            pr = [U_pr, V_pr]
            
            # Find indices of maxima
            mx_idx = [np.nanargmax(p) for p in pr]
            # Append y values of maxima to lists
            y_U.append(pr[0].y.values[mx_idx[0]])
            y_V.append(pr[1].y.values[mx_idx[1]])
            
            # Find wake width using specified method and append to lists
            mask1  = ~np.isnan(pr[0].values) # filter out nans
            sigma_U.append(wakewidth(wwm,y_to_ys(pr[0].y.to_numpy()[mask1],pr[0].values[mask1]),np.abs(pr[0].values[mask1])))
            
            mask2  = ~np.isnan(pr[1].values)
            sigma_V.append(wakewidth(wwm,y_to_ys(pr[1].y.to_numpy()[mask2],pr[1].values[mask2]),np.abs(pr[1].values[mask2])))
        
        # Convert lists to numpy arrays
        y_U = np.asarray(y_U)
        y_V = np.asarray(y_V)
        sigma_U = np.asarray(sigma_U)
        sigma_V = np.asarray(sigma_V)
        
        # Create figure and axes objects
        fig, ax = plt.subplots(1,1,figsize=FigureSize)
        
        # Plot wake centrelines
        l1, = ax.plot(x_Ds, y_U/self.D, label='$U$')
        l2, = ax.plot(x_Ds, y_V/self.D, label='$V$')
        vel_handles = [l1, l2]
        
        # Plot wake edges
        ax.plot(x_Ds, (y_U + sigma_U)/self.D, ls='--', c=list(mcolours)[0])
        ax.plot(x_Ds, (y_U - sigma_U)/self.D, ls='--', c=list(mcolours)[0])
        ax.plot(x_Ds, (y_V + sigma_V)/self.D, ls='--', c=list(mcolours)[1])
        ax.plot(x_Ds, (y_V - sigma_V)/self.D, ls='--', c=list(mcolours)[1])
        
        # Draw actuator disc
        draw_AD(ax, self.x_AD[0], self.y_AD[0], self.D/self.D, yaw)
        
        # Create legend entries for line styles
        solid_line = mlines.Line2D([], [], color='black', ls='-', label='Centreline')
        dashed_line = mlines.Line2D([], [], color='black', ls='--', label='Edges')
        
        # Add legends
        vel_legend = plt.legend(handles=vel_handles, loc='upper left')
        plt.gca().add_artist(vel_legend)
        ax.legend(handles=[solid_line, dashed_line], loc = 'lower left')
        
        # Add axes labels and limits
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title('$\gamma = %d^\circ$, $C_T = 1.0$, T.I.$= 0.1$, cells/D = %d' % (yaw, self.cD))
        ax.set_xlabel('$x/D$ [-]')
        ax.set_ylabel('$y/D$ [-]')
        
        # Save figure (.pdf and .svg)
        # fig.tight_layout()
        plt.show()
        filename = 'plot_wakeCentresandEdges_yaw_%d' % yaw
        fig.savefig('fig/' + filename + '.pdf',bbox_inches='tight')
        fig.savefig('fig/' + filename + '.svg',bbox_inches='tight')
        
def plot_velProf1(instances, var_lst, varname, pos, xlim=None, ylim=None, FigureSize=(6.4,4.8)):
    fig, ax = plt.subplots(1,1,figsize=FigureSize)
    
    varlabels = {
        'yaw':['$\gamma$','$^\circ$'],
        'ct':['$C_T$',''],
        'ti':['T.I.','']}
    
    for i, inst in enumerate(instances):
        # Extract profile
        V_pr = inst.V_zh.interp(x = inst.x_AD[0] + pos*inst.D)
        
        # Plot profile with label
        lab = varlabels.get(varname)[0] + '$= ' + str(var_lst[i]) + '$' + varlabels.get(varname)[1]
        ax.plot(V_pr.y/inst.D,V_pr,label=lab)
        
    # Add axes limits and labels
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('$y/D$ [-]')
    ax.set_ylabel('$V$ [m/s]')
    if varname == 'yaw':
        ax.set_title('$x/D = %d$, $C_T = 1.0$, T.I.$= 0.1$' % pos)
    elif varname == 'ct':
        ax.set_title('$x/D = %d$, $\gamma = 40^\circ$, T.I.$= 0.1$' % pos)
    elif varname == 'ti':
        ax.set_title('$x/D = %d$, $\gamma = 40^\circ$, $C_T = 1.0$' % pos)
    
    # Add legend
    plt.legend()
    
    # Save figure (.pdf and .svg)
    fig.tight_layout()
    filename = 'velProf1_%s' % varname
    fig.savefig('fig/' + filename + '.pdf',bbox_inches='tight')
    fig.savefig('fig/' + filename + '.svg',bbox_inches='tight')

def plot_LSS(instances, var_lst, varname, poss, wwm, xlim=None, ylim=None, FigureSize=(6.4,4.8)):
    wwm_labels = {
        'half-width': 'r_{1/2}',
        '1%'        : 'r_{1\%}',
        'Gaussian'  : '\sigma_y',
        'integral'  : '\sigma_y'
        }
    
    fig, axs = plt.subplots(2,2,figsize=FigureSize,sharex=True,sharey=True)
    
    axs = np.ravel(axs)
    
    for i, inst in enumerate(instances):
        for ip, pos in enumerate(poss):
            # Extract profile
            V_pr = inst.V_zh.interp(x = inst.x_AD[0] + pos*inst.D)
            
            # Calculate std
            mask  = ~np.isnan(V_pr.values)
            sigma = wakewidth(wwm, y_to_ys(V_pr.y.to_numpy()[mask], V_pr.values[mask]), np.abs(V_pr.values[mask]))
        
            # Plot profile with label
            lab = '$x/D = %d$' % pos
            if (varname == 'yaw') and (var_lst[i] == 0):
                axs[i].plot(V_pr.y.to_numpy()/sigma,V_pr/np.nanmax(V_pr),label=lab)
            else:
                # axs[i].plot(V_pr.y.to_numpy()/sigma,V_pr/np.nanmax(V_pr),label=lab)
                axs[i].plot(y_to_ys(V_pr.y.to_numpy(),V_pr.values)/sigma,V_pr/np.nanmax(V_pr),label=lab)
        
        # Add axes limits and labels
        axs[i].set_xlim(xlim)
        axs[i].set_ylim(ylim)
        if i > 1:
            axs[i].set_xlabel('$y*/%s$ [-]' % wwm_labels.get(wwm, 'r'))
        if i%2 == 0:
            axs[i].set_ylabel('$|V|/|V_{max}|$ [-]')
        if varname == 'yaw':
            axs[i].set_title('$\gamma = %d^\circ$, $C_T = 1.0$, T.I.$= 0.1$' % var_lst[i])
        elif varname == 'ct':
            axs[i].set_title('$\gamma = 40^\circ$, $C_T = %.1f$, T.I.$= 0.1$' % var_lst[i])
        elif varname == 'ti':
            axs[i].set_title('$\gamma = 40^\circ$, $C_T = 1.0$, T.I.$= %0.2f$' % var_lst[i])
            
        # Individual legends
        # axs[i].legend()
        
    # Add legend
    plt.legend(loc='center left', bbox_to_anchor=(1.025,1.1))
    
    # Save figure (.pdf and .svg)
    # fig.tight_layout()
    plt.show()
    filename = 'plot3_%s' % varname
    fig.savefig('fig/' + filename + '.pdf',bbox_inches='tight')
    fig.savefig('fig/' + filename + '.svg',bbox_inches='tight')
    
def plot_velProf_wr(wr, nwr, var_lst, varname, pos, xlim=None, ylim=None, FigureSize=(6.4,4.8)):
    fig, ax = plt.subplots(1,1,figsize=FigureSize)
    
    varlabels = {
        'yaw':['$\gamma$','$^\circ$'],
        'ct':['$C_T$',''],
        'ti':['T.I.','']}
    
    c = [wr, nwr]
    wr_plot = []
    nwr_plot = []
    
    for i in range(np.shape(c)[1]):
        # Extract profiles
        V_pr_wr  = c[0][i].V_zh.interp(x = c[0][i].x_AD[0] + pos*c[0][i].D)
        V_pr_nwr = c[1][i].V_zh.interp(x = c[1][i].x_AD[0] + pos*c[1][i].D)
        
        # Plot profile with label
        lab = varlabels.get(varname)[0] + '$= ' + str(var_lst[i]) + '$' + varlabels.get(varname)[1]
        l1, = ax.plot(V_pr_wr.y/c[0][i].D,V_pr_wr,label=lab,ls='-',c=list(mcolours)[i])
        wr_plot.append(l1)
        l2, = ax.plot(V_pr_nwr.y/c[1][i].D,V_pr_nwr,label=lab,ls='--',c=list(mcolours)[i])
        nwr_plot.append(l2)
    
    # Create legend entries for line styles
    solid_line = mlines.Line2D([], [], color='black', ls='-', label='On')
    dashed_line = mlines.Line2D([], [], color='black', ls='--', label='Off')
        
    # Add axes limits and labels
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('$y/D$ [-]')
    ax.set_ylabel('$V$ [m/s]')
    if varname == 'yaw':
        ax.set_title('$x/D = %d$, $C_T = 1.0$, T.I.$= 0.1$' % pos)
    elif varname == 'ct':
        ax.set_title('$x/D = %d$, $\gamma = 40^\circ$, T.I.$= 0.1$' % pos)
    elif varname == 'ti':
        ax.set_title('$x/D = %d$, $\gamma = 40^\circ$, $C_T = 1.0$' % pos)
    
    # Add legends
    yaw_legend = plt.legend(handles=wr_plot, loc='upper left')
    plt.gca().add_artist(yaw_legend)
    ax.legend(handles=[solid_line, dashed_line], loc = 'upper right', title='Wake rotation')
    
    # Save figure (.pdf and .svg)
    fig.tight_layout()
    filename = 'velProf_wr_%s' % varname
    fig.savefig('fig/' + filename + '.pdf',bbox_inches='tight')
    fig.savefig('fig/' + filename + '.svg',bbox_inches='tight')
    
def plot_LSS_wr(wr, nwr, var_lst, varname, poss, wwm, xlim=None, ylim=None, FigureSize=(6.4,4.8)):
    wwm_labels = {
        'half-width': 'r_{1/2}',
        '1%'        : 'r_{1\%}',
        'Gaussian'  : '\sigma_y',
        'integral'  : '\sigma_y'
        }
    
    fig, axs = plt.subplots(2,2,figsize=FigureSize,sharex=True,sharey=True)
    
    axs = np.ravel(axs)
    
    c = [wr, nwr]
    
    for i in range(np.shape(c)[1]):
        wr_plot = []
        nwr_plot = []
        for ip, pos in enumerate(poss):
            # Extract profiles
            V_pr_wr  = c[0][i].V_zh.interp(x = c[0][i].x_AD[0] + pos*c[0][i].D)
            V_pr_nwr = c[1][i].V_zh.interp(x = c[1][i].x_AD[0] + pos*c[1][i].D)
            
            # Calculate stds
            mask1  = ~np.isnan(V_pr_wr.values)
            sigma1 = wakewidth(wwm, y_to_ys(V_pr_wr.y.to_numpy()[mask1], V_pr_wr.values[mask1]), np.abs(V_pr_wr.values[mask1]))
            
            mask2  = ~np.isnan(V_pr_nwr.values)
            sigma2 = wakewidth(wwm, y_to_ys(V_pr_nwr.y.to_numpy()[mask2], V_pr_nwr.values[mask2]), np.abs(V_pr_nwr.values[mask2]))
        
            # Plot profile with label
            lab = '$x/D = %d$' % pos
            if (varname == 'yaw') and (var_lst[i] == 0):
                l1, = axs[i].plot(V_pr_wr.y.to_numpy()/sigma1,V_pr_wr/np.nanmax(V_pr_wr),label=lab,ls='-',c=list(mcolours)[ip])
                l2, = axs[i].plot(V_pr_nwr.y.to_numpy()/sigma2,V_pr_nwr/np.nanmax(V_pr_nwr),label=lab,ls='--',c=list(mcolours)[ip])
                wr_plot.append(l1)
                nwr_plot.append(l2)
            else:
                # axs[i].plot(V_pr.y.to_numpy()/sigma,V_pr/np.nanmax(V_pr),label=lab)
                l1, = axs[i].plot(y_to_ys(V_pr_wr.y.to_numpy(),V_pr_wr.values)/sigma1,V_pr_wr/np.nanmax(V_pr_wr),label=lab,ls='-',c=list(mcolours)[ip])
                l2, = axs[i].plot(y_to_ys(V_pr_nwr.y.to_numpy(),V_pr_nwr.values)/sigma2,V_pr_nwr/np.nanmax(V_pr_nwr),label=lab,ls='--',c=list(mcolours)[ip])
                wr_plot.append(l1)
                nwr_plot.append(l2)
        
        # Add axes limits and labels
        axs[i].set_xlim(xlim)
        axs[i].set_ylim(ylim)
        if i > 1:
            axs[i].set_xlabel('$y*/%s$ [-]' % wwm_labels.get(wwm, 'r'))
        if i%2 == 0:
            axs[i].set_ylabel('$|V|/|V_{max}|$ [-]')
        if varname == 'yaw':
            axs[i].set_title('$\gamma = %d^\circ$, $C_T = 1.0$, T.I.$= 0.1$' % var_lst[i])
        elif varname == 'ct':
            axs[i].set_title('$\gamma = 40^\circ$, $C_T = %.1f$, T.I.$= 0.1$' % var_lst[i])
        elif varname == 'ti':
            axs[i].set_title('$\gamma = 40^\circ$, $C_T = 1.0$, T.I.$= %0.2f$' % var_lst[i])
            
        # Individual legends
        # axs[i].legend()
    
    # Create legend entries for line styles
    solid_line = mlines.Line2D([], [], color='black', ls='-', label='On')
    dashed_line = mlines.Line2D([], [], color='black', ls='--', label='Off')    
    
    # Add legends
    yaw_legend = plt.legend(handles=wr_plot, loc='center left', bbox_to_anchor=(1.025,1.7))
    plt.gca().add_artist(yaw_legend)
    axs[i].legend(handles=[solid_line, dashed_line], loc = 'center left', bbox_to_anchor=(1.025, 0.5), title='Wake rotation')
    
    # Save figure (.pdf and .svg)
    # fig.tight_layout()
    plt.show()
    filename = 'plot_LSS_wr_%s' % varname
    fig.savefig('fig/' + filename + '.pdf',bbox_inches='tight')
    fig.savefig('fig/' + filename + '.svg',bbox_inches='tight')

def wakeDomain(var,x_wd,y_wd):
    return var.where((var.x >= x_wd[0]) & (var.x <= x_wd[1]) & (var.y >= y_wd[0]) & (var.y <= y_wd[1]))

def draw_AD(ax,x,y,D,theta):
    '''
    Draw an actuator disc (AD) at (x,y) rotated theta degrees anticlockwise about its centre.

    Parameters
    ----------
    ax : axes._subplots.AxesSubplot
        Axes on which the AD is to be drawn.
    x : float
        x-coordinate of AD centre.
    y : float
        y-coordinate of AD centre.
    D : float
        Diameter of AD - units should match that of plot.
    theta : float
        Rotation of AD in degrees. 

    Returns
    -------
    None.

    '''
    R = D/2
    theta = theta*np.pi/180
    # Upper point of rotor
    x1 = np.sin(theta)*R
    y1 = np.cos(theta)*R
    # Lower point of rotor
    x2 = -x1
    y2 = -y1
    # Plot
    ax.plot([x+x1,x+x2],[y+y1,y+y2],'k')
    
def y_to_ys(y,V):
    '''
    

    Parameters
    ----------
    y : Array of float64
        Original spanwise coordinates.
    V : Array of float64
        Lateral velocities corresponding to original spanwise coordinates, y.

    Returns
    -------
    ys : Array of float64
        New spanwise coordinates with ys = 0 at maximum V.

    '''
    
    idx = np.nanargmax(np.abs(V))
    ys  = y - y[idx]
    
    return ys

def wakewidth(method,ys,V):
    '''
    Calculate width of wake profile using one of methods given in dictionary.

    Parameters
    ----------
    method : string
        Name of method used to calculate characteristic wake width.
    ys : Array of float64
        Lateral coordinates with Vmax at y*=0.
    V : Array of float64
        Lateral velocities at coordinates ys.

    Returns
    -------
    sigma : float
        Wake width [same units as ys].

    '''
    methods = {
        'half-width': [xwidth, 0.5],
        '1%'        : [xwidth, 0.01],
        'Gaussian'  : [fit_Gaussian],
        'integral'  : [integral]
        }
    # Get the function from the switcher dictionary
    func = methods.get(method, 0)[0]
    # Execute the function
    if method == 'integral':
        sigma = func(ys,V)
    elif method == 'Gaussian':
        amp, mu, sigma = func(ys,V)
    else:
        x     = methods.get(method, 0)[1]
        sigma = func(ys,V,x)
    return sigma
    

def xwidth(ys,V,x):
    '''
    Calculate

    Parameters
    ----------
    ys : Array of float64
        Lateral coordinates with Vmax at y*=0.
    V : Array of float64
        Lateral velocities at coordinates ys.

    Raises
    ------
    Exception
        If velocities are not monotonic (increasing or decreasing) then np.interp() will not work as expected, therefore raise exception.

    Returns
    -------
    r_half : float
        Wake half-width - average width where V = 0.5*V_max. Averaging takes account of skewed profiles.

    '''
    # calculate half of maximum velocity
    V_x      = np.max(V) * x
    
    # split velocities either side of y* = 0
    lower    = ys <= 0
    upper    = ys >= 0
    V_lower  = V[lower]
    V_upper  = -V[upper]
    Vs       = [V_lower, V_upper]
    
    # check whether velocities are monotonic as they approach y* = 0
    for V in Vs:
        mono = np.all(np.diff(V) > 0)
        if mono == False:
            raise Exception('Velocities are not monotonic. Use a different wake width method.')
    # Calculate average half width for each side and average in case profile is skewed
    r_x = [-sp.interp1d(V_lower,ys[lower],bounds_error=True)(V_x)]
    r_x.append(sp.interp1d(V_upper,ys[upper],bounds_error=True)(-V_x))
    r_x = np.mean(r_x)
    
    return r_x

def fit_Gaussian(ys,V):
    popt, pcov = curve_fit(Gauss, ys, V, p0=[np.max(V),0,250])
    
    # fit_y = Gauss(ys, popt[0], popt[1], popt[2]) # amp, mu, sigma
    
    # fig1, ax1 = plt.subplots(1,1)

    # ax1.plot(ys/popt[2], V/np.max(V), 'o', label='data')
    # ax1.plot(ys/popt[2], fit_y/np.max(V), '-', label='Gaussian fit')
    # ax1.set_xlabel('$y^*/\sigma$ [-]')
    # ax1.set_ylabel('$|V|/|V_{max}|$ [-]')
    # ax1.legend()
    # plt.show()
    return popt

def integral(ys,V):
    return 1/(np.sqrt(2*np.pi)*np.max(V)) * spi.simps(V,ys)

def Gauss(x, amp, mu, sigma):
    return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))

#%% Other plots - Plot planes of V at hub height for different yaw angles
# if len(yaws) > 1:
#     fig, axs = plt.subplots(len(yaws),1,figsize=FigureSize,sharex=True)
#     plt.subplots_adjust(hspace=0.3)
    
#     for iy, yaw in enumerate(yaws):
#         lab = r'$\gamma = ' + str(yaw) + r'^\circ$'
#         p = axs[iy].contourf(V[iy].x/D,V[iy].y/D,V[iy].T,vmin=Vmin[iy],vmax=Vmax[iy])
#         u.draw_AD(axs[iy],x_AD[0],y_AD[0],D/D,yaw)
#         axs[iy].set_xlim([-2, 10]); axs[iy].set_ylim([-2, 2])
#         axs[iy].set_ylabel('$y/D$ [-]')
#         axs[iy].text(0.98,0.92,lab,horizontalalignment='right',verticalalignment='top',transform=axs[iy].transAxes,color='1')
    
#     axs[iy].set_xlabel('$x/D$ [-]')
#     # divider = make_axes_locatable(ax)
#     fig.subplots_adjust(right=0.8)
#     # cax = divider.append_axes("right", size = "5%", pad=0.05)
#     cax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#     cbar = fig.colorbar(p, ax=axs.ravel(), cax=cax)
#     cbar.set_label('$V$ [m/s]')
#     plt.show()
    
#     fig.savefig('V_zh_yaws.pdf',bbox_inches='tight')
#     fig.savefig('V_zh_yaws.svg',bbox_inches='tight')