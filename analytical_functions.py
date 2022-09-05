# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 11:32:06 2022

@author: nilsg
"""

import numpy as np
import xarray as xr
import warnings

# Suppress FutureWarnings (for xarray)
warnings.simplefilter(action='ignore', category=FutureWarning)

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
    flowdata_rec = flowdata.interp(x=x_rotor, y=y_rotor, z=z_rotor)

    # Filter rotor disc from rectangle
    flowdata_rotor = flowdata_rec.where(np.sqrt((flowdata_rec["r"] - y_t)**2 + (flowdata_rec.z - z_t)**2) <= (D/2),
                             other=np.nan)
    
    # Create disc averaged velocity vector (assuming W=0)
    vel_d = np.asarray([flowdata_rotor.U.mean(), flowdata_rotor.V.mean(), 0])
    
    # Create disc normal vector (pointing downstream)
    n = np.asarray([np.cos(yaw), -np.sin(yaw), 0])
    
    # Dot product disc velocity vector with rotor normal vector
    vel_AD = np.dot(vel_d, n)
    
    return vel_AD

def turb_add(y, z, y1, z1, yaw1, s, CT, y2, z2, yaw2, D, sep_x, I0):
    '''
    

    Parameters
    ----------
    y : Array of float64
        y-coordinates of flow domain.
    z : Array of float64
        z-coordinates of flow domain.
    y1 : float
        y-coordinate of upstream turbine's wake centre.
    z1 : float
        z-coordinate of upstream turbine's wake centre.
    yaw1 : float
        Yaw angle of upstream turbine [rad].
    s : float
        Width of wake of upstream turbine at x-position of downstream position.
    CT : float
        Thrust coefficient of upstream turbine.
    y2 : float
        y-coordinate of downstream turbine.
    z2 : float
        z-coordinate of downstream turbine.
    yaw1 : float
        Yaw angle of downstream turbine [rad].
    D : float
        Rotor diameter of both turbines.
    sep_x : float
        x-separation of two turbines.
    I0 : float
        Freestream turbulence intensity.

    Returns
    -------
    H : float
        Contribution to added turbulence of turbine 1 on turbine 2.

    '''
    if sep_x < 0.1: # because I_p tends to infinity as separation tends to zero
        H = 0
    else:
        # axial induction factor of turbine 1
        a = 0.5 * (1 - np.sqrt(1 - CT*np.cos(yaw1))) 
        # Added turbulence intensity due to turbine 1 according to modified Crespo model (see Bastankhah et al., 2021)
        I_p = 0.66 * a**0.83 * I0**0.03 * (sep_x/D)**(-0.32)
        
        ## Radii within yz-plane at turbine 2 x-position
        # turbine 1 (upstream) wake diameter
        rw = 2 * s # why 2*sigma? see Porté-Agel et al., 2013, "A numerical..."
        # turbine 2 (downstream) rotor diameter
        r2 = D/2
        
        ## Overlap area
        if (yaw1 == 0) & (yaw2 == 0):
            # Rotor centre separation in yz-plane at turbine 2 x-position
            sep_yz = np.sqrt((y1-y2)**2 + (z1-z2)**2)
            
            # If turbine 2 rotor not within turbine 1 wake
            if sep_yz >= (rw + r2):
                A = 0 # overlap area = zero
            # If turbine 2 fully enveloped by turbine 1 wake
            elif sep_yz <= (rw - r2):
                A = np.pi * r2**2 # overlap area = rotor area
            # If there is partial overlap
            else:
                # Use formula for partial overlap of two circles with different radii
                d1 = (rw**2 - r2**2 + sep_yz**2) / (2*sep_yz)
                d2 = sep_yz - d1
                A  = rw**2*np.arccos(d1/rw) - d1*np.sqrt(rw**2-d1**2) + r2**2*np.arccos(d2/r2) - d2*np.sqrt(r2**2-d2**2)
                
        else:
            # Meshgrid y and z coordinates of plane
            Y, Z = np.meshgrid(y, z)
    
            # Create mask for turbine 2 rotor
            in_rotor = (((Y - y2)/(r2*np.cos(yaw2)))**2 + ((Z - z2)/r2)**2) <= 1
            # Create mask for turbine 1 wake
            in_wake  = (((Y - y1)/(rw*np.cos(yaw1)))**2 + ((Z - z1)/rw)**2) <= 1
            # Create mask for overlap
            in_both  = in_rotor & in_wake
            
            # Count number of points within masks
            n_rotor = np.count_nonzero(in_rotor)
            n_both  = np.count_nonzero(in_both)
            
            # Overlap area
            A = (n_both/n_rotor) * np.pi * r2**2
        
        # Contribution of turbulence intensity (see Niayifar & Porté-Agel, 2016, Eq. 18)
        H = ((A * 4) / (np.pi * D**2)) * I_p
        
    return H

def epsilon(CT):
    beta = 0.5 * (1 + np.sqrt(1-CT)) / (np.sqrt(1-CT))
    e    = 0.2 * np.sqrt(beta)
    return e

def NREL5MW(U_d):
    # PyWakeEllipSys
    CT_curve = np.asarray([[3.0,  0.0  ],
                          [4.0,  0.913],
                          [5.0,  0.869],
                          [6.0,  0.780],
                          [7.0,  0.773],
                          [8.0,  0.770],
                          [9.0,  0.768],
                          [10.0, 0.765],
                          [11.0, 0.746],
                          [12.0, 0.533],
                          [13.0, 0.392],
                          [14.0, 0.305],
                          [15.0, 0.244],
                          [16.0, 0.200],
                          [17.0, 0.167],
                          [18.0, 0.142],
                          [19.0, 0.122],
                          [20.0, 0.106],
                          [21.0, 0.092],
                          [22.0, 0.082],
                          [23.0, 0.073],
                          [24.0, 0.065],
                          [25.0, 0.059]])
    
    P_curve = np.asarray([[3.0,  0.0  ],
                          [4.0,  209.2],
                          [5.0,  444.8],
                          [6.0,  765.2],
                          [7.0,  1212.0],
                          [8.0,  1807.7],
                          [9.0,  2571.6],
                          [10.0, 3523.8],
                          [11.0, 4649.7],
                          [12.0, 5000.0],
                          [13.0, 5000.0],
                          [14.0, 5000.0],
                          [15.0, 5000.0],
                          [16.0, 5000.0],
                          [17.0, 5000.0],
                          [18.0, 5000.0],
                          [19.0, 5000.0],
                          [20.0, 5000.0],
                          [21.0, 5000.0],
                          [22.0, 5000.0],
                          [23.0, 5000.0],
                          [24.0, 5000.0],
                          [25.0, 5000.0]])

    CP_curve = ((P_curve[:,1]*1e3/(1-0.059)) / (0.125 * 1.225 * 126.0**2 * np.pi * P_curve[:,0]**3))

    CP_curve = np.vstack((P_curve[:,0], CP_curve)).T
    
    # SOWFA
    # CT_curve = np.asarray([[2.0,  1.4612728464576872],
    #                     [2.5,  1.3891500248600195],
    #                     [3.0,  1.268082754962957 ],
    #                     [3.5,  1.1646999475504172],
    #                     [4.0,  1.0793803926905128],
    #                     [4.5,  1.0098020917279509],
    #                     [5.0,  0.9523253671258429],
    #                     [5.5,  0.9048200632193146],
    #                     [6.0,  0.8652746358037285],
    #                     [6.5,  0.8317749797630494],
    #                     [7.0,  0.8032514305647592],
    #                     [7.5,  0.7788892341777304],
    #                     [8.0,  0.7730863447173755],
    #                     [8.5,  0.7726206761501038],
    #                     [9.0,  0.7721934195205071],
    #                     [9.5,  0.7628473779358198],
    #                     [10.0, 0.7459330274762097],
    #                     [10.5, 0.7310049480450205],
    #                     [11.0, 0.7177914274917664],
    #                     [11.5, 0.799361832581412 ],
    #                     [12.0, 0.8871279360742889],
    #                     [12.5, 0.9504655842078242],
    #                     [13.0, 1.0000251651970853],
    #                     [13.5, 1.0390424010487957],
    #                     [14.0, 1.0701572223736   ],
    #                     [14.5, 1.0945877239199593]])
    
    # SOWFA
    # CP_curve = np.asarray([[2.0,  -0.2092219804533027],
    #                     [2.5,  0.2352391893638198 ],
    #                     [3.0,  0.46214453324002824],
    #                     [3.5,  0.5476677311380832 ],
    #                     [4.0,  0.5772456648046942],
    #                     [4.5,  0.5833965967255043],
    #                     [5.0,  0.5790298877294793],
    #                     [5.5,  0.5701467792599509],
    #                     [6.0,  0.5595564940228319],
    #                     [6.5,  0.5480479331210222],
    #                     [7.0,  0.5366246493538858],
    #                     [7.5,  0.5258303873334416],
    #                     [8.0,  0.5229191014420005],
    #                     [8.5,  0.5224657416437077],
    #                     [9.0,  0.5220516710065948],
    #                     [9.5,  0.5175531496262384],
    #                     [10.0, 0.5092952304943719],
    #                     [10.5, 0.5016730194861562],
    #                     [11.0, 0.4946298748497652],
    #                     [11.5, 0.5326349577484786],
    #                     [12.0, 0.5597671514540806],
    #                     [12.5, 0.5679550280111124],
    #                     [13.0, 0.5659876382489049],
    #                     [13.5, 0.5572755521043566],
    #                     [14.0, 0.5441595739848516],
    #                     [14.5, 0.5280326705762761]])
    
    CT = np.interp(U_d, CT_curve[:,0], CT_curve[:,1])
    CP = np.interp(U_d, CP_curve[:,0], CP_curve[:,1])
    
    return CT, CP