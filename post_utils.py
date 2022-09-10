# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 11:14:50 2022

@author: nilsg
"""

import numpy as np
import xarray
import warnings
import matplotlib.pyplot as plt
import os
import scipy.integrate as spi
from scipy.optimize import curve_fit
from lateralSolution import lateralSolution
from streamwiseSolution import streamwiseSolution
import time
from analytical_functions import vel_disc, NREL5MW

# Suppress FutureWarnings (for xarray)
warnings.simplefilter(action='ignore', category=FutureWarning)

class windTurbine():
    def __init__(self, D, zh, TSR=None, CP=0.47):
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
        self.x_wt = np.asarray(x_wt) # WT x coordinates (original)
        self.y_wt = np.asarray(y_wt) # WT y coordinates (original)
        self.wts  = wts  # list of wind turbines
        self.n_wt = len(wts) # number of wind turbines
        self.yaws = np.deg2rad(np.asarray(yaws)) # yaw angles of turbines (follows order of wts)
        self.CTs  = CTs  # thrust coefficients [-]
        
        # Default parameters
        if CTs is None:
            self.CTs = [0.8]*self.n_wt
        if wrs is None:
            self.wrs  = [True]*self.n_wt
        
        # Wind farm generalised turbine parameters (NOT ROBUST)
        self.zh   = self.wts[0].zh # hub height [m]
        self.D    = self.wts[0].D  # rotor diameter [m]
        
        self.z_wt = [self.zh] * self.n_wt # WT z coordinates
        
        # Wind farm coordinates (PyWakeEllipSys convention)
        self.x_AD, self.y_AD = get_PWE_coords(self.x_wt, self.y_wt)
        self.z_AD = self.zh * np.ones(np.shape(self.x_AD))
    
class flowcase():
    def __init__(self, dir_path, wf, cD, shift=True):
        self.dir_path = dir_path # flowdata.nc directory path
        self.wf       = wf       # wf instance of flowcase
        self.cD       = cD       # cells per diameter
        
        # Load flow data upon initialisation
        self.load_flowdata()
        self.load_flowdata_remove()
        self.flowdata_diff(shift=shift)
    
    def load_flowdata(self):
        # Load full 3D flow data as xarray.Dataset
        self.flowdata = xarray.open_dataset(self.dir_path + 'flowdata.nc')
        
        # Define useful variables
        self.U    = self.flowdata.U
        self.U_zh = self.flowdata.U.interp(z=self.wf.zh)
        self.V    = self.flowdata.V
        self.V_zh = self.flowdata.V.interp(z=self.wf.zh)
        self.U_total = np.sqrt(self.flowdata.U**2 + self.flowdata.V**2 + self.flowdata.W**2)
        self.U_total_zh = self.U_total.interp(z=self.wf.zh)
    
    def load_flowdata_remove(self):
        # Load flow data with turbines removed
        self.flowdata_remove = [0] * self.wf.n_wt
        for i_wt in range(self.wf.n_wt):
            dir_path = self.dir_path + 'remove/' + str(i_wt)
            if os.path.exists(dir_path):
                self.flowdata_remove[i_wt] = xarray.open_dataset(dir_path + '/flowdata.nc')
    
    def flowdata_diff(self, shift):
        self.flowdata_diff = [0] * self.wf.n_wt
        self.Udef          = [0] * self.wf.n_wt
        self.Udef_zh       = [0] * self.wf.n_wt
        self.Vdef          = [0] * self.wf.n_wt
        self.Vdef_zh       = [0] * self.wf.n_wt
        for i_wt in range(self.wf.n_wt):
            if self.flowdata_remove[i_wt] != 0:
                if shift:
                    # Shift flowdata.variables such that first turbine is at x = 0
                    # Create x_wt_r and y_wt_r and then calculate the shift required to convert them to x_AD_r and y_AD_r then shift by this amount
                    x_wt_r = np.delete(self.wf.x_wt, i_wt)
                    y_wt_r = np.delete(self.wf.y_wt, i_wt)
                    
                    # Calculate difference between original wf centre and wf centre with turbine removed
                    x_shift = (np.ptp(x_wt_r) - np.ptp(self.wf.x_wt)) / 2
                    y_shift = (np.ptp(y_wt_r) - np.ptp(self.wf.y_wt)) / 2
                    
                    # Shift flowdata
                    self.flowdata_remove[i_wt] = self.flowdata_remove[i_wt].shift(x = int(x_shift*self.cD), y = int(y_shift*self.cD))
                
                # Calculate deficits
                self.flowdata_diff[i_wt] = self.flowdata_remove[i_wt] - self.flowdata
                
                # Define useful variables
                self.Udef[i_wt] = self.flowdata_diff[i_wt].U
                self.Udef_zh[i_wt] = self.Udef[i_wt].interp(z=self.wf.zh)
                self.Vdef[i_wt] = -self.flowdata_diff[i_wt].V
                self.Vdef_zh[i_wt] = self.Vdef[i_wt].interp(z=self.wf.zh)
    
    def velocityProfile(self, velComp, pos, WT):
        if all([isinstance(x, int) for x in self.Udef]):
            profiles = {
                'U' : self.U_zh.interp(x = self.wf.x_AD[WT]*self.wf.D + pos*self.wf.D),
                'V' : self.V_zh.interp(x = self.wf.x_AD[WT]*self.wf.D + pos*self.wf.D),
                'Udef' : self.wf.Uinf - self.U_zh.interp(x = self.wf.x_AD[WT]*self.wf.D + pos*self.wf.D)
                }
        else:
            profiles = {
                'U' : self.U_zh.interp(x = self.wf.x_AD[WT]*self.wf.D + pos*self.wf.D),
                'V' : self.V_zh.interp(x = self.wf.x_AD[WT]*self.wf.D + pos*self.wf.D),
                'Udef' : self.Udef_zh[WT].interp(x = self.wf.x_AD[WT]*self.wf.D + pos*self.wf.D),
                'Vdef' : self.Vdef_zh[WT].interp(x = self.wf.x_AD[WT]*self.wf.D + pos*self.wf.D)
                }
        return profiles.get(velComp)
    
    def analytical_solution(self, method, near_wake_correction, removes = [], U_h=None, V_h=0, rho=1.225):
        '''
        Computes streamwise and lateral velocity fields for flowcase using analytical solution.

        Parameters
        ----------
        method : string
            Version of the conservation of momentum deficit to be used, either "original" or "modified". See Bastankhah et al., 2021, for further details.
        U_h : float, optional
            Hub height streamwise velocity. By default this is interpolated from the inflow profile in PyWakeEllipSys, so may not be exactly equal to the value specified in the PyWakeEllipSys run script.
        V_h : float, optional
            Hub height lateral velocity. The default is 0.
        rho : float, optional
            Air density [kg/m^3]. The default is 1.225.

        Returns
        -------
        U : Array of float64 (nx,ny,nz)
            Streamwise velocity field.
        V : Array of float64 (nx,ny,nz)
            Lateral velocity field.

        '''
        # Define turbine z positions
        z_t = self.wf.zh * np.ones((self.wf.n_wt))

        # Sort turbine x, y, z positions by increasing x
        idx = np.argsort(self.wf.x_wt)
        x_t = self.wf.x_wt[idx]
        y_t = self.wf.y_wt[idx]
        z_t = z_t[idx]
        yaws = self.wf.yaws[idx]

        ## Flow domain
        x = self.flowdata.x.values
        y = self.flowdata.y.values
        z = self.flowdata.z.values

        nx, ny, nz = len(x), len(y), len(z)

        ## Inflow
        # Preallocate analytical velocity field
        U0 = np.zeros((nx, ny, nz)) # streamwise velocity
        V0 = np.zeros((nx, ny, nz)) # lateral velocity
        
        # Convert from total T.I. to streamwise T.I.
        I0 = 0.8 * self.wf.ti

        # Preallocate streamwise velocity field with flowcase inflow
        U_in = self.flowdata.U.values[0,0,:]
        if U_h is None:
            U_h  = np.interp(self.wf.zh, self.flowdata.z.values, U_in)

        U0[:, :, :] = U_in

        ## Deficit
        # Preallocate velocities for removed turbines
        flowdata_remove = [0] * self.wf.n_wt

        ## Run solution
        start_time = time.time() # start timer
        flowdata, P = lateralSolution(method, self.wf.n_wt, x_t, y_t, z_t, yaws, x, y, z, U0, U_h, V0, I0, near_wake_correction, rho, self.wf.zh, self.wf.D)
        for i_t in removes:
            # remove turbine from layout
            x_t_r = np.delete(self.wf.x_wt, i_t)
            y_t_r = np.delete(self.wf.y_wt, i_t)
            z_t_r = self.wf.zh * np.ones((self.wf.n_wt-1))
            # remove value from yaws
            yaws_r = np.delete(yaws, i_t)
            # run solution
            flowdata_remove[i_t], _ = lateralSolution(method, self.wf.n_wt-1, x_t_r, y_t_r, z_t_r, yaws_r, x, y, z, U0, U_h, V0, I0, near_wake_correction, rho, self.wf.zh, self.wf.D)
        end_time = time.time() # end timer
        
        ## Calculate deficits
        flowdata_def = [0] * self.wf.n_wt
        for i_t in range(self.wf.n_wt):
            if flowdata_remove[i_t] != 0:
                flowdata_def[i_t] = flowdata_remove[i_t] - flowdata

        ## Timing
        execution_time = end_time - start_time
        mins = execution_time // 60
        secs = execution_time % 60
        print("Solution execution time: {:.0f}m {:.1f}s".format(mins, secs))
        
        return flowdata, flowdata_def, P, U_h
    
    def streamwiseSolution(self, method, removes = [], U_h=None, rho=1.225):
        '''
        Computes streamwise and lateral velocity fields for flowcase using analytical solution.

        Parameters
        ----------
        method : string
            Version of the conservation of momentum deficit to be used, either "original" or "modified". See Bastankhah et al., 2021, for further details.
        U_h : float, optional
            Hub height streamwise velocity. By default this is interpolated from the inflow profile in PyWakeEllipSys, so may not be exactly equal to the value specified in the PyWakeEllipSys run script.
        rho : float, optional
            Air density [kg/m^3]. The default is 1.225.

        Returns
        -------
        U : Array of float64 (nx,ny,nz)
            Streamwise velocity field.

        '''
        # Define turbine z positions
        z_t = self.wf.zh * np.ones((self.wf.n_wt))

        # Sort turbine x, y, z positions by increasing x
        idx = np.argsort(self.wf.x_wt)
        x_t = self.wf.x_wt[idx]
        y_t = self.wf.y_wt[idx]
        z_t = z_t[idx]
        yaws = self.wf.yaws[idx]

        ## Flow domain
        x = self.flowdata.x.values
        y = self.flowdata.y.values
        z = self.flowdata.z.values

        nx, ny, nz = len(x), len(y), len(z)

        ## Inflow
        # Preallocate analytical velocity field
        U0 = np.zeros((nx, ny, nz)) # streamwise velocity

        # Preallocate streamwise velocity field with flowcase inflow
        U_in = self.flowdata.U.values[0,0,:]
        if U_h is None:
            U_h  = np.interp(self.wf.zh, self.flowdata.z.values, U_in)

        U0[:, :, :] = U_in

        ## Deficit
        # Preallocate velocities for removed turbines
        flowdata_remove = [0] * self.wf.n_wt

        ## Run solution
        start_time = time.time() # start timer
        flowdata, P = streamwiseSolution(method, self.wf.n_wt, x_t, y_t, z_t, yaws, x, y, z, U0, U_h, self.wf.ti, rho, self.wf.zh, self.wf.D)
        for i_t in removes:
            # remove turbine from layout
            x_t_r = np.delete(self.wf.x_wt, i_t)
            y_t_r = np.delete(self.wf.y_wt, i_t)
            z_t_r = self.wf.zh * np.ones((self.wf.n_wt-1))
            # remove value from yaws
            yaws_r = np.delete(yaws, i_t)
            # run solution
            flowdata_remove[i_t], _ = streamwiseSolution(method, self.wf.n_wt-1, x_t_r, y_t_r, z_t_r, yaws_r, x, y, z, U0, U_h, self.wf.ti, rho, self.wf.zh, self.wf.D)
        end_time = time.time() # end timer
        
        ## Calculate deficits
        flowdata_def = [0] * self.wf.n_wt
        for i_t in range(self.wf.n_wt):
            if flowdata_remove[i_t] != 0:
                flowdata_def[i_t] = flowdata_remove[i_t] - flowdata

        ## Timing
        execution_time = end_time - start_time
        mins = execution_time // 60
        secs = execution_time % 60
        print("Solution execution time: {:.0f}m {:.1f}s".format(mins, secs))
        
        return flowdata, flowdata_def, P, U_h
    
    def powerAndThrust(self, rho=1.225):
        P = np.zeros((self.wf.n_wt))
        T = np.zeros((self.wf.n_wt))
        U_d = np.zeros((self.wf.n_wt))
        for i_t in range(self.wf.n_wt):
            # Calculate disc velocity
            U_d[i_t] = vel_disc(self.flowdata, 
                           self.wf.x_wt[i_t]*self.wf.D, 
                           self.wf.y_wt[i_t]*self.wf.D, 
                           self.wf.z_wt[i_t], 
                           self.wf.yaws[i_t],
                           self.wf.D)
            
            ###### CORRECTION HERE FOR SOME ERROR WITH U_d THAT I DON'T HAVE TIME TO FIX BEFORE THE HAND-IN ########
            
            # Look up CT and CP
            CT, CP = NREL5MW(U_d[i_t])
            
            # Calculate thrust force from CT
            T[i_t] = (np.pi * rho * CT * U_d[i_t]**2 * self.wf.D**2) / 8
            
            # Calculate power from CP
            P[i_t] = (np.pi * rho * CP * U_d[i_t]**3 * self.wf.D**2) / 8
            
        return U_d, T
            
def get_PWE_coords(x_wt, y_wt):
    x_AD = x_wt - np.ptp(x_wt)/2
    y_AD = y_wt - np.ptp(y_wt)/2
    return x_AD, y_AD
            
def draw_AD(ax,view,x,y,D,yaw):
    '''
    Draw an actuator disc (AD) at (x,y) rotated yaw radians anticlockwise about its centre.

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
        Rotation of AD in radians. 

    Returns
    -------
    None.

    '''
    R = D/2
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
        theta = np.linspace(0, 2*np.pi, 100)
        a = R*np.cos(yaw)*np.cos(theta)
        b = R*np.sin(theta)
        ax.plot(x+a,y+b,'k')
    if view == 'side':
        theta = np.linspace(0, 2*np.pi, 100)
        a = R*np.sin(yaw)*np.cos(theta)
        b = R*np.sin(theta)
        ax.plot(x+a,y+b,'k')
    
def y_to_ys(y,vel,method):
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
    # idx = np.nanargmax(np.abs(vel)) # peak
    y_c, _ = wakeCentre(method, y, vel)
    
    ys  = y - y_c
    
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
    if method == 'unyawed':
        vel_c = 0
        
        idx_max = np.argmax(vel)
        idx_min = np.argmin(vel)

        if np.argmax(vel) < np.argmin(vel):
            vel = vel[idx_max:idx_min]
            y = y[idx_max:idx_min]
        else:
            vel = vel[idx_min:idx_max]
            y = y[idx_min:idx_max]

        y_c = np.interp(vel_c, y, vel)        
        
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
    sigma = 1/(np.sqrt(2*np.pi)*vel_max) * spi.simps(np.abs(vel),y)
    
    return sigma

def Gaussian(x, amp, mu, sigma):
    return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))

# def Gaussian(x, mu, sigma):
    # return (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-(x - mu)**2 / (2 * sigma**2))

def set_size(width, fraction=1, height_adjust=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio
    # Adjust figure height
    fig_height_in = fig_height_in * height_adjust

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim