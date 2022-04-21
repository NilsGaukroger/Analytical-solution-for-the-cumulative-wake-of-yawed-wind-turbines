# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 14:24:57 2022

@author: nilsg
"""

import numpy as np
import xarray
import warnings
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import copy
import scipy.integrate as spi
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors
import matplotlib.lines as mlines

# Create object for default matplotlib line colours
mcolours = mcolors.TABLEAU_COLORS

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class windTurbine():
    def __init__(self, D, zh, TSR, CP=0.47):
        self.D   = D   # rotor diameter [m]
        self.zh  = zh  # hub height [m]
        self.CP  = CP  # power coefficient [-]
        self.TSR = TSR # tip-speed ratio [-]
        
    def rpm(self, Uinf, yaw):
        '''
        

        Parameters
        ----------
        Uinf : float
            Inflow wind speed at hub height.
        yaw : float
            Turbine yaw angle.

        Returns
        -------
        N : float
            Turbine rotational speed in rpm.

        '''
        # Calculate velocity normal to rotor plane
        U_normal = Uinf * np.cos(np.deg2rad(yaw))
        # Calculate rotational speed
        N = ((self.TSR * U_normal) / (self.D/2)) * (30/np.pi)
        return N
    
class windFarm():
    def __init__(self, Uinf, ti, x_wt, y_wt, wts, yaws=None, CTs=None, wrs=None):
        self.Uinf = Uinf # inflow velocity at hub height
        self.ti   = ti   # inflow turbulence intensity @ zh
        self.x_wt = x_wt # WT x coordinates (original)
        self.y_wt = y_wt # WT y coordinates (original)
        self.wts  = wts  # list of wind turbines
        self.yaws = yaws # yaw angles of turbines (follows order of wts)
        self.CTs  = CTs  # thrust coefficients [-]
        self.wrs  = wrs  # wake rotations
        
        # Default parameters
        if CTs is None:
            self.CTs = [0.8]*len(wts)
        if wrs is None:
            self.wrs  = [True]*len(wts)
        
        # Wind farm generalised turbine parameters (NOT ROBUST)
        self.zh   = self.wts[0].zh # hub height [m]
        self.D    = self.wts[0].D  # rotor diameter [m]
        
        # Wind farm coordinates (PyWakeEllipSys convention)
        self.x_AD, self.y_AD = self.get_PWE_coords()
        self.z_AD = self.zh * np.ones(np.shape(self.x_AD))
    
    def get_PWE_coords(self):
        self.x_AD = np.array(self.x_wt) - (self.x_wt[-1]-self.x_wt[0])/2
        self.y_AD = np.array(self.y_wt)
        return self.x_AD, self.y_AD
    
class flowcase():
    def __init__(self, infile, wf):
        self.infile = infile # flowdata.nc filepath
        self.wf     = wf     # wf instance of flowcase
        
        # Load flow data upon initialisation
        self.load_flowdata()
    
    def load_flowdata(self):
        # Load full 3D flow data as xarray.Dataset
        self.flowdata = xarray.open_dataset(self.infile)
        
        # Define useful variables
        self.U    = self.flowdata.U
        self.U_zh = self.flowdata.U.interp(z=self.wf.zh)
        self.V    = self.flowdata.V
        self.V_zh = self.flowdata.V.interp(z=self.wf.zh)
    
    def velocityProfile(self, velComp, pos):
        profiles = {
            'U' : self.wf.Uinf - self.U_zh.interp(x = self.wf.x_AD[0] + pos*self.wf.D),
            'V' : self.V_zh.interp(x = self.wf.x_AD[0] + pos*self.wf.D)
            }
        return profiles.get(velComp)
    
    def vel_disc(self, velComp, n, plot=False, FigureSize=None, xlim=None, ylim=None):
        # Choose velocity component
        velComps = {
            'U' : self.U,
            'V' : self.V
            }
        vel = velComps.get(velComp)
        
        # Convert turbine number to index
        n = n - 1
        
        ## Initialise totals
        vel_d = 0 # total contribution of all sampled points
        ns    = 0 # total number of sampled points
        
        # For all points in yz-plane at x == x_t
        for j in range(len(vel.y)):
            for k in range(len(vel.z)):
                in_disc = (((vel.y[j] - self.wf.y_AD[n])/((self.wf.wts[n].D/2)*np.cos(self.wf.yaws[n])))**2 + ((vel.z[k] - self.wf.z_AD[n])/(self.wf.wts[n].D/2))**2) <= 1
                if in_disc:
                    vel_d = vel_d + np.interp(self.wf.x_AD[n], vel.x, vel[:,j,k])
                    ns    = ns + 1 # add to point counter
        
        if plot:
            fig, ax = plt.subplots(1,1,figsize=FigureSize)
            
            Y, Z = np.meshgrid(vel.y, vel.z)
            p = ax.contourf(Y/self.wf.D, Z/self.wf.D, vel.interp(x=self.wf.x_AD[n]).T)
            draw_AD(ax, 'front', self.wf.y_AD[n]/self.wf.wts[n].D, self.wf.wts[n].zh/self.wf.D, self.wf.wts[n].D/self.wf.D, self.wf.yaws[n])
            ax.set_xlabel('$y/D$ [-]')
            ax.set_ylabel('$z/D$ [-]')
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            cbar = fig.colorbar(p)
            cbar.set_label('${:s}$ [m/s]'.format(velComp))
            
            plt.show()
        
        return vel_d / ns # average disc velocity
    
    def plot_contourWithProfiles(self, poss, xlim=None, ylim=(-3,3), levels=100, FigureSize=None):
        # Plot plane of V at hub height for single case with profiles at several downstream positions
        
        # Set default xlim according to poss
        if xlim is None:
            xlim = (-2, np.max(poss)+1)
        
        fig, axs = plt.subplots(2, len(poss), figsize=FigureSize, sharey=True)
        gs = axs[0,0].get_gridspec() # get geometry of subplot grid
        
        # Remove top row of subplots
        for ax in axs[0,:]:
            ax.remove()
            
        # Replace with a single plot spanning the whole width
        axbig = fig.add_subplot(gs[0,:])
        
        # Plot contour of lateral velocity at hub height
        p = axbig.contourf(self.V_zh.x/self.wf.D, self.V_zh.y/self.wf.D, self.V_zh.T, levels=levels)
        
        # Add actuator disc to plot
        draw_AD(axbig, 'top', 0, 0, self.wf.D/self.wf.D, self.wf.yaws[0])
        
        # Set axes limits
        axbig.set_xlim(xlim); axbig.set_ylim(ylim)
        
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
            V_pr = self.velocityProfile('V', pos)
            
            # Plot profiles
            if self.wf.yaws[0] == 0: # plot against y
                axs[1,ip].plot(V_pr, V_pr.y/self.wf.D)
            else: # plot against y*
                axs[1,ip].plot(V_pr, y_to_ys(V_pr.y, V_pr)/self.wf.D)
                
            # Add axes limits and labels
            axs[1,ip].set_ylim(ylim)
            axs[1,ip].set_xlabel('$V$ [m/s]')
            
            # Label each plot with x/D
            lab = r'$%.0fD$' % pos
            axs[1,ip].text(0.98, 0.95, lab, horizontalalignment='right', verticalalignment='top', transform=axs[1,ip].transAxes)
            
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
            
        if self.wf.yaws[0] == 0:
            axs[1,0].set_ylabel('$y/D$ [-]')
        else:
            axs[1,0].set_ylabel('$y_V^*/D$ [-]')
        plt.suptitle('$V_{{z_h}}$ for $\gamma = {:d}^\circ$, $C_T = {:.1f}$, T.I.$={:.0f}\%$, W.R.$=$ {:s}'.format(self.wf.yaws[0], self.wf.CTs[0], self.wf.ti*100, str(self.wf.wrs[0])))
        
        # Save figure (.pdf and .svg)
        fig.tight_layout()
        filename = 'contourWithProfiles'
        figpath  = 'fig/yaw/' + str(self.wf.yaws[0]) + '/'
        if not os.path.exists(figpath): # if the directory doesn't exist create it
            os.makedirs(figpath)
        fig.savefig(figpath + filename + '.pdf', bbox_inches='tight')
        fig.savefig(figpath + filename + '.svg', bbox_inches='tight')
    
    def plot_SS_fitGaussian(self, poss, wcm, wwm, ylim=(0,None), xlim=None, FigureSize=None):
        
        # Create subplots object
        fig, ax = plt.subplots(1,1,figsize=FigureSize)
        
        # Preallocate arrays for fitting parameters
        amps = np.empty((len(poss)))
        mus  = np.empty((len(poss)))
        sigs = np.empty((len(poss)))
        
        for ip, pos in enumerate(poss):
            # Extract profile
            V_pr = self.velocityProfile('V', pos)
            
            # Calculate wake centre velocity
            _, V_c = wakeCentre(wcm, V_pr.y, V_pr)
            
            # Calculate std
            sigma = wakeWidth(wwm, V_pr.y, V_pr)
            
            # Plot profile with label
            lab = '$x/D = {:d}$'.format(pos)
            ax.plot(y_to_ys(V_pr.y, V_pr)/sigma, np.abs(V_pr)/V_c, 'o', label=lab)
            
            # Fit Gaussian
            if wwm == 'Gaussian' or 'integral':
                amps[ip], mus[ip], sigs[ip], _ = fit_Gaussian(V_pr.y, V_pr/V_c)
                amp = np.mean(amps)
                mu  = np.mean(mus)
                sig = np.mean(sigs)
            
        # Plot average fitted Gaussian
        ax.plot(y_to_ys(V_pr.y, V_pr)/sig, Gaussian(V_pr.y, amp, mu, sig), c='k', ls='--', label='Gaussian')
            
        # Add axes limits and labels
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('$y^*_V/\sigma_{y,V}$ [-]')
        ax.set_ylabel('$V/V_c$ [-]')
        ax.set_title('$\gamma = {:d}^\circ$, $C_T = {:.1f}$, T.I.$={:.0f}\%$, W.R.$=${:s}'.format(self.wf.yaws[0], self.wf.CTs[0], self.wf.ti*100, str(self.wf.wrs[0])))
        
        # Add legend
        ax.legend()
        
        # Save figure (.pdf and .svg)
        plt.show()
        filename = 'SS_fitGaussian'
        figpath  = 'fig/yaw/' + str(self.wf.yaws[0]) + '/'
        if not os.path.exists(figpath): # if the directory doesn't exist create it
            os.makedirs(figpath)
        fig.savefig(figpath + filename + '.pdf', bbox_inches='tight')
        fig.savefig(figpath + filename + '.svg', bbox_inches='tight')
    
    def plot_wakeCentreAndEdges(self, poss, wcm, wwm, xlim=None, ylim=None, FigureSize=None):
        # Create lists for wake centres and wake widths
        wc_U = np.empty((len(poss)))
        wc_V = np.empty((len(poss)))
        ww_U = np.empty((len(poss)))
        ww_V = np.empty((len(poss)))
        
        for i, pos in enumerate(poss):
            # Extract profiles
            U_pr = self.velocityProfile('U', pos)
            V_pr = self.velocityProfile('V', pos)
            
            # Find wake centres
            wc_U[i], _ = wakeCentre(wcm, U_pr.y, U_pr)
            wc_V[i], _ = wakeCentre(wcm, V_pr.y, V_pr)
            
            # Find wake width and append to lists
            ww_U[i] = wakeWidth(wwm, U_pr.y, U_pr)
            ww_V[i] = wakeWidth(wwm, V_pr.y, V_pr)
            
        # Create subplots object
        fig, ax = plt.subplots(1, 1, figsize=FigureSize)
        
        # Add line at y=0
        y0_line = ax.axhline(y=0, xmin=0, xmax=1, ls=':', c='k', label='$y=0$')
        
        # Plot wake centrelines
        l1, = ax.plot(poss, wc_U/self.wf.D, ls='-', label='$U$')
        l2, = ax.plot(poss, wc_V/self.wf.D, ls='-', label='$V$')
        vel_handles = [l1, l2]
        
        # Plot wake edges
        ax.plot(poss, (wc_U+ww_U)/self.wf.D, ls='--', c=list(mcolours)[0])
        ax.plot(poss, (wc_U-ww_U)/self.wf.D, ls='--', c=list(mcolours)[0])
        ax.plot(poss, (wc_V+ww_V)/self.wf.D, ls='--', c=list(mcolours)[1])
        ax.plot(poss, (wc_V-ww_V)/self.wf.D, ls='--', c=list(mcolours)[1])
        
        # Draw actuator disc
        draw_AD(ax, 'top', self.wf.x_AD[0], self.wf.y_AD[0], self.wf.D/self.wf.D, self.wf.yaws[0])
        
        # Create legend entries for line styles
        centreline  = mlines.Line2D([], [], color='black', ls='-', label='Centreline')
        edges = mlines.Line2D([], [], color='black', ls='--', label='Edges')
        
        # Add legends
        vel_legend = plt.legend(handles=vel_handles, loc='upper left')
        plt.gca().add_artist(vel_legend)
        ax.legend(handles=[centreline, edges, y0_line], loc='lower left')
        
        # Add axes labels and limits
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title('$\gamma = {:d}^\circ$, $C_T = {:.1f}$, T.I.$= {:.0f}\%$'.format(self.wf.yaws[0], self.wf.CTs[0], self.wf.ti*100))
        ax.set_xlabel('$x/D$ [-]')
        ax.set_ylabel('$y/D$ [-]')
        
        # Save figure (.pdf and .svg)
        plt.show()
        filename = 'wakeCentreAndEdges'
        figpath  = 'fig/yaw/' + str(self.wf.yaws[0]) + '/'
        if not os.path.exists(figpath): # if the directory doesn't exist create it
            os.makedirs(figpath)
        fig.savefig(figpath + filename + '.pdf', bbox_inches='tight')
        fig.savefig(figpath + filename + '.svg', bbox_inches='tight')
        
class flowcaseGroup():
    def __init__(self, var, vals, path, wf, flowcases=None):
        if flowcases is not None:
            self.flowcases = flowcases
        
        # Load flowdata for all cases in group
        self.load_flowdata(var, vals, path, wf)
    
    def load_flowdata(self, var, vals, path, wf_template):
        self.var  = var  # variable: 'yaw', 'ct', 'ti', or 'wr'
        self.vals = vals # values of variable
        self.path = path # path to data directories
        
        self.flowcases = []
        for i, val in enumerate(vals):
            wf = copy.copy(wf_template)
            wf.wts = copy.copy(wf_template.wts)
            infile = path + var + '/' + str(val) + '/flowdata.nc'
            self.flowcases.append(flowcase(infile, wf))
            if self.var == 'yaw':
                self.flowcases[i].wf.yaws = [val]
            if self.var == 'ct':
                self.flowcases[i].wf.CTs = [val]
            if self.var == 'ti':
                self.flowcases[i].wf.ti = val
            if self.var == 'wr':
                self.flowcases[i].wf.wrs   = [val]
                self.flowcases[i].wf.yaws = [val]
                self.flowcases[i].load_flowdata()
            self.flowcases[i].load_flowdata()
            
    def plot_VvelocityProfiles(self, pos, xlim=None, ylim=None, FigureSize=None):
        # Create labels
        varlabels = {
            'yaw':['$\gamma$','$^\circ$'],
            'ct':['$C_T$',''],
            'ti':['T.I.','']
            }
        
        # Create subplots object
        fig, ax = plt.subplots(1, 1, figsize=FigureSize)
        
        for i, flowcase in enumerate(self.flowcases):
            # Extract profile
            V_pr = flowcase.velocityProfile('V', pos)
            
            # Plot profile with label
            lab = varlabels.get(self.var)[0] + '$= ' + str(self.vals[i]) + '$' + varlabels.get(self.var)[1]
            ax.plot(V_pr.y/flowcase.wf.D, V_pr, label=lab)
        
        # Add axes limits and labels
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('$y/D$ [-]')
        ax.set_ylabel('$V$ [m/s]')
        
        # Add plot title
        titles = {
            'yaw' : r'$x/D = {:d}$, $C_T = {:.1f}$, T.I.$= {:.0f}$%, W.R.$=${:s}'.format(pos, flowcase.wf.CTs[0], flowcase.wf.ti*100, str(flowcase.wf.wrs[0])),
            'ct'  : r'$x/D = {:d}$, $\gamma = {:d}^\circ$, T.I.$= {:.0f}$%, W.R.$=${:s}'.format(pos, flowcase.wf.yaws[0], flowcase.wf.ti*100, str(flowcase.wf.wrs[0])),
            'ti'  : r'$x/D = {:d}$, $\gamma = {:d}^\circ$, $C_T = {:.1f}$, W.R.$=${:s}'.format(pos, flowcase.wf.yaws[0], flowcase.wf.CTs[0], str(flowcase.wf.wrs[0]))
            }
        ax.set_title(titles.get(self.var))
        
        # Add legend
        ax.legend()
        
        # Save figure (.pdf and .svg)
        fig.tight_layout()
        plt.show()
        filename = 'VvelocityProfiles_{:s}'.format(self.var)
        figpath  = 'fig/' + self.var + '/'
        if not os.path.exists(figpath): # if the directory doesn't exist create it
            os.makedirs(figpath)
        fig.savefig(figpath + filename + '.pdf', bbox_inches='tight')
        fig.savefig(figpath + filename + '.svg', bbox_inches='tight')
        
    def plot_LSS(self, poss, wcm, wwm, xlim=None, ylim=None, FigureSize=None):
        # Create subplots object
        fig, axs = plt.subplots(2, 2, figsize=FigureSize, sharex=True, sharey=True)
        
        # Make axes indexing 1D
        axs = np.ravel(axs)
        
        for i, flowcase in enumerate(self.flowcases):
            for ip, pos in enumerate(poss):
                # Extract profile
                V_pr = flowcase.velocityProfile('V', pos)
                
                # Calculate wake centre velocity
                _, V_c = wakeCentre(wcm, V_pr.y, V_pr)
                
                # Calculate wake width
                sigma = wakeWidth(wwm, V_pr.y, V_pr)
                
                # Plot profile with label
                lab = '$x/D = {:d}$'.format(pos)
                if flowcase.wf.yaws[0] == 0:
                    axs[i].plot(V_pr.y/sigma, V_pr, label=lab)
                else:
                    axs[i].plot(y_to_ys(V_pr.y, V_pr)/sigma, V_pr/V_c, label=lab)
            
            # Add axes limits and labels
            axs[i].set_xlim(xlim)
            axs[i].set_ylim(ylim)
            if i > 1:
                axs[i].set_xlabel('$y^*_V/\sigma_{y,V}$')
            if i%2 == 0:
                axs[i].set_ylabel('$V/V_c$ [-]')
            titles = {
                'yaw' : r'$\gamma = {:d}^\circ$'.format(flowcase.wf.yaws[0]),
                'ct'  : r'$C_T = {:.1f}$'.format(flowcase.wf.CTs[0]),
                'ti'  : r'T.I.$={:.0f}\%$'.format(flowcase.wf.ti*100)
                }
            suptitles = {
                'yaw' : r'$C_T = {:.1f}$, T.I.$= {:.0f}$%'.format(flowcase.wf.CTs[0], flowcase.wf.ti*100),
                'ct' : r'$\gamma = {:d}^\circ$, T.I.$= {:.0f}$%'.format(flowcase.wf.yaws[0], flowcase.wf.ti*100),
                'ti' : r'$\gamma = {:d}^\circ$, $C_T = {:.1f}$'.format(flowcase.wf.yaws[0], flowcase.wf.CTs[0])
                }
            axs[i].set_title(titles.get(self.var))
            fig.suptitle(suptitles.get(self.var))
        
        # Add legend
        plt.legend(loc='center left', bbox_to_anchor=(1.025,1.1))
        
        # Save figure (.pdf and .svg)
        # fig.tight_layout()
        plt.show()
        filename = 'LSS'
        figpath  = 'fig/' + self.var + '/'
        if not os.path.exists(figpath): # if the directory doesn't exist create it
            os.makedirs(figpath)
        fig.savefig(figpath + filename + '.pdf', bbox_inches='tight')
        fig.savefig(figpath + filename + '.svg', bbox_inches='tight')
    
    # def plot_VvelocityProfiles_wr()
            
def draw_AD(ax,view,x,y,D,yaw):
    '''
    Draw an actuator disc (AD) at (x,y) rotated yaw degrees anticlockwise about its centre.

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
    yaw = yaw*np.pi/180
    if view == 'top':
        # Upper point of rotor
        x1 = np.sin(yaw)*R
        y1 = np.cos(yaw)*R
        # Lower point of rotor
        x2 = -x1
        y2 = -y1
        # Plot
        ax.plot([x+x1,x+x2],[y+y1,y+y2],'k')
    if view == 'front':
        xp = np.linspace(x-R,x+R,100)
        yp = y + np.sqrt(R**2 - xp**2)
        ax.plot(xp,yp,'k')
    
def y_to_ys(y,vel):
    '''
    

    Parameters
    ----------
    y : Array of float64
        Original spanwise coordinates.
    vel : Array of float64
        Velocities corresponding to original spanwise coordinates, y.

    Returns
    -------
    ys : Array of float64
        New spanwise coordinates with ys = 0 at maximum vel.

    '''
    
    # Convert inputs to numpy arrays
    y   = np.array(y)
    vel = np.array(vel)
    
    # Find wake centre
    idx = np.nanargmax(np.abs(vel)) # peak
    # idx = wakeCentre(y, vel)
    
    ys  = y - y[idx]
    
    return ys

def wakeCentre(method, y, vel):
    # Convert inputs to numpy arrays
    y   = np.array(y)
    vel = np.array(vel)
    
    # Take magnitude of vel
    vel = np.abs(vel)
    
    ## Find wake centre and velocity
    if method == 'max':
        idx   = np.argmax(vel)
        y_c   = y[idx]
        vel_c = vel[idx]
    if method == 'Gaussian':
        # fit a Gaussian and use mean as index
        vel_c, y_c, _, _ = fit_Gaussian(y, vel)
    return y_c, vel_c

def wakeWidth(method, y, vel):
    # Define switcher dictionary
    methods = {
        'Gaussian'  : [fit_Gaussian],
        'integral'  : [integral]
        }
    
    # Get function from dictionary
    func = methods.get(method, 0)[0]
    
    # Execute the function
    if method == 'integral':
        sigma = func(y,vel)
    if method == 'Gaussian':
        amp, mu, sigma, pcov = func(y,vel)
    
    return sigma

def fit_Gaussian(y, vel, p0=None, plot=False, FigureSize=None):
    # Convert inputs to numpy arrays
    y   = np.array(y)
    vel = np.array(vel)
    # Take magnitude of vel
    vel = np.abs(vel)
    # Set default initial guess values for fitting
    if p0 is None:
        p0 = [np.max(vel), 0, 100]
        # p0 = [0, -100]
    # Find fitted parameters and covariance
    popt, pcov = curve_fit(Gaussian, y, vel, p0=p0)
    amp   = popt[0]
    mu    = popt[1]
    sigma = popt[2]
    # mu    = popt[0]
    # sigma = popt[1]
    
    # Plot fit
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=FigureSize)
        ax.plot(y, vel, label='data')
        ax.plot(y, Gaussian(y, amp, mu, sigma), label='fit', ls='--')
        ax.axvline(mu, 0, 1, ls='--', c='k', label='$\mu$')
        ax.axvline(mu+sigma, 0, 1, ls='-.', c='k', label='$1\sigma$')
        ax.axvline(mu-sigma, 0, 1, ls='-.', c='k')
        ax.set_xlim((mu-5*sigma, mu+5*sigma))
        plt.legend()
        plt.show()
    
    return amp, mu, sigma, pcov
    
def integral(y,vel):   
    # Find peak velocity
    vel_max, _, _, _ = fit_Gaussian(y, vel)
    
    # Calculate wake width
    sigma = 1/(np.sqrt(2*np.pi)*vel_max) * spi.simps(vel,y)
    
    return sigma

def Gaussian(x, amp, mu, sigma):
    return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))

# def Gaussian(x, mu, sigma):
    # return (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-(x - mu)**2 / (2 * sigma**2))