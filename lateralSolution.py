# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 11:10:39 2022

@author: nilsg
"""

def lateralSolution(method, n_t, x_t, y_t, z_t, yaws, x, y, z, U0, U_h, V0, I0, rho=1.225, zh=90.0, D=126.0):
    '''
    Analytical solution for the cumulative wake of yawed wind turbines. Yields streamwise and lateral velocity fields for a given turbine type, wind farm layout and set of yaw angles.

    Parameters
    ----------
    method : string
        Version of the conservation of momentum deficit to be used, either "original" or "modified". See Bastankhah et al., 2021, for further details.
    n_t : int
        Number of turbines, should match len(x_t) = len(y_t) = len(z_t).
    x_t : Array of float64
        x-coordinates of turbines.
    y_t : Array of float64
        y-coordinates of turbines.
    z_t : Array of float64
        z-coordinates of turbines.
    yaws : Array of int32
        Yaw angles of turbines in radians, positive is clockwise rotation when viewed from above.
    x : Array of float64
        x-coordinates of cells within flow domain.
    y : Array of float64
        y-coordinates of cells within flow domain.
    z : Array of float64
        z-coordinates of cells within flow domain.
    U0 : Array of float64 (nx,ny,nz)
        Initial streamwise velocity field.
    V0 : Array of float64 (nx,ny,nz)
        Initial lateral velocity field.
    rho : float, optional
        Air density [kg/m^3]. The default is 1.225.
    zh : float, optional
        Hub height [m]. The default is 90.0.
    D : float, optional
        Rotor diameter [m]. The default is 126.0.

    Returns
    -------
    U : Array of float64 (nx,ny,nz)
        Streamwise velocity field.
    V : Array of float64 (nx,ny,nz)
        Lateral velocity field.

    '''

    #%% Module imports
    import numpy as np
    from analytical_functions import vel_disc, NREL5MW, turb_add, epsilon
    from numpy.lib.scimath import sqrt as csqrt
    from copy import copy
    import xarray
    from scipy.interpolate import interp1d
    
    #%% Setup checks
    # Sort turbine x, y, z positions by increasing x
    idx = np.argsort(x_t)
    x_t = x_t[idx]
    y_t = y_t[idx]
    z_t = z_t[idx]
    
    # Initialise velocity fields
    U = copy(U0)
    V = copy(V0)
    
    # Make velocities into xarray DataArray
    flowdata = xarray.Dataset(
        data_vars = dict(
            U  = (["x","y","z"], U),
            V  = (["x","y","z"], V),
            U0 = (["x","y","z"], U0),
            V0 = (["x","y","z"], V0)
        ),
        coords = dict(
            x = ("x", x),
            y = ("y", y),
            z = ("z", z)),
        attrs = dict(description = "Analytical solution"),
    )
    
    #%% Flow domain
    nx = len(x)
    Y, Z = np.meshgrid(y, z)
    Y = Y.T; Z = Z.T
    
    #%% Analytical solution for streamwise velocity
    # Preallocation
    U_d  = np.zeros((n_t))
    CT   = np.zeros((n_t))
    CP   = np.zeros((n_t))
    T    = np.zeros((n_t))
    P    = np.zeros((n_t))
    I    = np.zeros((n_t))
    H    = np.zeros((n_t))
    k_w  = np.zeros((n_t)) # Streamwise wake recovery
    k_w2 = np.zeros((n_t)) # Lateral wake recovery
    s    = np.zeros((n_t)) # Streamwise wake width
    s2   = np.zeros((n_t)) # Lateral wake width
    
    C    = np.zeros((n_t, nx), dtype=np.complex_) # Streamwise
    C2   = np.zeros((n_t, nx), dtype=np.complex_) # Lateral
    
    # Initialise deflection
    delta_D = np.zeros((n_t, nx))
    
    print('Computing velocity field...')
    for n in range(n_t):
        #%% Thrust force
        # Calculate rotor-averaged streamwise velocity
        U_d[n] = vel_disc(flowdata, x_t[n]*D, y_t[n]*D, z_t[n], yaws[n], D)
        # Look up corresponding thrust and power coefficients for NREL 5MW
        CT[n], CP[n] = NREL5MW(U_d[n])
        # Calculate thrust force from CT
        T[n] = (np.pi * rho * CT[n] * U_d[n]**2 * D**2) / 8
        # Calculate power from CP
        P[n] = (np.pi * rho * CP[n] * U_d[n]**3 * D**2) / 8
        
        #%% Turbulence intensity
        if n == 0:
            # For first turbine, use inflow turbulence intensity
            I[n] = I0
        else:
            # For all other turbines, include added turbulence from other turbines
            for i in range(n):
                # Added T.I. of all other turbines on turbine n
                H[i] = turb_add(y,
                                z,
                                (y_t[i] + np.interp(x_t[n]*D,x,delta_D[i,:]))*D,
                                z_t[i],
                                yaws[i],
                                k_w[i] * (x_t[n] - x_t[i])*D + epsilon(CT[i]) * D,
                                CT[i],
                                y_t[n]*D,
                                z_t[n],
                                yaws[n],
                                D,
                                (x_t[n] - x_t[i])*D,
                                I0)
            # Total T.I. at turbine n
            I[n] = np.sqrt(I0**2 + max(H[:n], default=0)**2)
        
        #%% Wake recovery rate
        # k_w[n] = 0.38 * I[n] # Niayifar & Porté-Agel, 2016
        # k_w[n] = 0.35 * I[n] # Carbajo Fuertes et al., 2018
        # k_w[n] = 0.34 * I[n] # Zhan et al., 2020
        # k_w[n] = 0.31 * I[n] # Bastankhah et al., 2021
        k_w[n] = 0.4 * I[n] # Gaukroger, 2022
        k_w2[n] = 0.6 * I[n] # Gaukroger, 2022
        
        #%% Velocity
        # For all streamwise points in CFD domain
        downstream = x > x_t[n]*D
        first      = np.argmax(downstream)
        x_n        = x[downstream]
        
        # Interpolate for centreline velocities
        U0_c = flowdata.U0.interp(y=y_t[n]*D, z=z_t[n])
        V0_c = flowdata.V0.interp(y=y_t[n]*D, z=z_t[n])
        Uc_int = interp1d(x, U0_c)
        Vc_int = interp1d(x, V0_c)
        
        for i in range(len(x_n)):
            # Add offset to x-coordinate
            i = i + first
            #%% Sum of lambda * C_i/U_h
            # Compute wake width of turbine n at position x[i]
            s[n] = k_w[n] * (x[i] - x_t[n]*D) + epsilon(CT[n]) * D # streamwise
            s2[n] = k_w2[n] * (x[i] - x_t[n]*D) + epsilon(CT[n]) * D # lateral
            # Set total for sum of lambda * C_i/U_h from (4.10) = 0
            L = 0; L1 = 0; L2 = 0
            if n > 0: # because lambda = 0 for the first turbine?
                # Compute sum of lambda * C_i/U_h term from (4.10)
                for ii in range(n-1, -1, -1):
                    s[ii] = k_w[ii] * (x[i] - x_t[ii]*D) + epsilon(CT[ii]) * D
                    l_dict = {
                        "original" : ((2*s[ii]**2)/(s[n]**2 + s[ii]**2)) * np.exp(-((((y_t[n] - (y_t[ii] + delta_D[ii,i]))*D)**2)/(2*(s[n]**2 + s[ii]**2)))) * np.exp(-(((z_t[n] - z_t[ii])**2)/(2*(s[n]**2 + s[ii]**2)))),
                        "modified" : ((s[ii]**2)/(s[n]**2 + s[ii]**2)) * np.exp(-((((y_t[n] - (y_t[ii] + delta_D[ii,i]))*D)**2)/(2*(s[n]**2 + s[ii]**2)))) * np.exp(-(((z_t[n] - z_t[ii])**2)/(2*(s[n]**2 + s[ii]**2))))
                        }
                    l = l_dict.get(method)
                    # Add contribution to sum
                    L += l * (np.real(C[ii,i]))
                    # Lateral
                    s2[ii] = k_w2[ii] * (x[i] - x_t[ii]*D)
                    l1 = ((2*s2[ii]**2)/(s[n]**2 + s2[ii]**2)) * np.exp(-((((y_t[n] - y_t[ii])*D)**2)/(2*(s[n]**2 + s2[ii]**2)))) * np.exp(-(((z_t[n] - z_t[ii])**2)/(2*(s[n]**2 + s2[ii]**2))))
                    # Add contribution to sum
                    L1 += l1 * np.real(C2[ii,i])
                    l2 = ((2*s[ii]**2)/(s2[n]**2 + s[ii]**2)) * np.exp(-((((y_t[n] - y_t[ii])*D)**2)/(2*(s2[n]**2 + s[ii]**2)))) * np.exp(-(((z_t[n] - z_t[ii])**2)/(2*(s2[n]**2 + s[ii]**2))))
                    # Add contribution to sum
                    L2 += l2 * np.real(C[ii,i])
            
            #%% Solve for Cn
            # New proposed model
            C[n,i] = (U_h-L) * (1 - csqrt(1 - ((T[n] * np.cos(yaws[n])) / (rho * np.pi * s[n]**2 * (U_h-L)**2) )))
            # Linear superposition with local velocity as reference velocity
            # C[n,i] = U_d[n] * (1 - np.sqrt(1 - ( CT[n] / (8 * s[n]**2/D**2) )))
            # Near wake region
            if ((np.imag(C[n,i] != 0)) or ((C[n,i]/U_d[n]) > (1 - np.sqrt(1-CT[n])))):
                C[n,i] = (1 - np.sqrt(1-CT[n])) * U_d[n]
            
            #%% Solve for C2n
            # Coupled solution
            C2[n,i] = (((T[n] * np.sin(yaws[n])) / (np.pi * rho)) + C[n,i] * s[n]**2 * L1) / (s2[n]**2 * (2*U_h - L2))
            
            # Near wake correction
            # if C2[n,i]/U_h > 0.08:
            #     C2[n,i] = 0.08 * U_h
            
            # Decoupled solution
            # C2[n,i] = (T[n] * np.sin(yaws[n])) / (2 * np.pi * rho * s2[n]**2 * U_h)
            
            #%% Solve for Un through domain using (4.5)
            # Streamwise
            f = np.exp(-(((Y - (y_t[n] + delta_D[n,i])*D)**2)/(2*s[n]**2))) * np.exp(-(((Z - z_t[n])**2)/(2*s[n]**2)))
            flowdata.U[i,:,:] += -np.real(C[n,i]) * f
            
            # Lateral
            f2 = np.exp(-(((Y - y_t[n]*D)**2)/(2*s2[n]**2))) * np.exp(-(((Z - z_t[n])**2)/(2*s2[n]**2)))
            flowdata.V[i,:,:] += np.real(C2[n,i]) * f2
            
            #%% Compute wake centre deflection
            # Wake centre velocities (option 1)
            Uc = Uc_int(x[i]) - np.real(C[n,i])
            Vc = Vc_int(x[i]) + np.real(C2[n,i])
            
            # Wake centre velocities (option 2)
            # Uc = interpolate.interpn((x,y,z), U, (x[i], y_t[n]*D, z_t[n]))
            # Vc = interpolate.interpn((x,y,z), V, (x[i], y_t[n]*D, z_t[n]))
            
            # Deflection
            if i < len(x_n)+first-1:
                # delta_D[n,i+1] = delta_D[n,i] + ((Vc/U_h)*(x[i+1]-x[i]))/D
                # delta_D[n,i+1] = delta_D[n,i] + ((Vc/Uc)*(x[i+1]-x[i]))/D
                if (x[i] - x_t[n]*D) > 5*D:
                    delta_D[n,i+1] = delta_D[n,i] + ((Vc/Uc)*(x[i+1]-x[i]))/D
            
        #%% Status check
        print('turbine {:d} of {:d} ({:.0f}% complete)'
              .format(n+1, n_t, ((n+1)/n_t)*100))
    
    return flowdata, P

#%% if __name__ = '__main__':
if __name__ == '__main__':
    #%% Module imports
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    from scipy.interpolate import interp1d
    
    def vel_zh(z, zh, U, V):
        U_int = interp1d(z, U, axis=2)
        V_int = interp1d(z, V, axis=2)
        U_zh  = U_int(zh)
        V_zh  = V_int(zh)
        return U_zh, V_zh    
    
    #%% Setup
    # Layout alignment
    alignment = 'aligned' # 'aligned' or 'slanted'

    # Choice of method
    method = "original" # "original" or "modified" (see Bastankhah et al., 2020)

    # Define constants
    rho = 1.225 # air density [kg/m^3]
    zh  = 90.0  # turbine hub height [m]
    D   = 126.0 # turbine rotor diameter [m]

    # Define number of turbines
    n_t = 15

    # Define layout parameters
    s_x     = 7    # streamwise inter-turbine spacing [D]
    s_y     = 4    # lateral inter-turbine spacing [D]
    stagger = 0.75 # stagger [D]

    # Define turbine x and y positions
    x_t = np.empty((n_t))
    y_t = np.empty((n_t))
    for i_t in range(n_t):
        x_t[i_t] = np.floor(i_t/3) * s_x
        if alignment == 'aligned':
            y_t[i_t] = ((i_t-1)%3) * s_y - s_y
        elif alignment == 'slanted':
            y_t[i_t] = ((i_t-1)%3) * s_y + np.floor(i_t/3) * stagger - s_y
        else:
            print('Incorrect alignment specification.')
            break

    # Plot layout        
    fig, ax = plt.subplots(1,1)
    ax.scatter(x_t,y_t)
    ax.set_xlabel('x/D')
    ax.set_ylabel('y/D')

    # Define turbine z positions
    z_t = zh * np.ones((n_t))

    # Define yaw angles
    yaws = np.deg2rad(np.asarray([25]*n_t))

    # Sort turbine x, y, z positions by increasing x
    idx = np.argsort(x_t)
    x_t = x_t[idx]
    y_t = y_t[idx]
    z_t = z_t[idx]
    yaws = yaws[idx]

    print('Setup complete.')
    
    #%% Flow domain
    x = np.linspace(min(x_t)-2, max(x_t)+10,  100)*D
    y = np.linspace(min(y_t)-1,  max(y_t)+1,  100)*D
    z = np.linspace(         0,           2,  100)*zh
    
    nx, ny, nz = len(x), len(y), len(z)
    
    #%% Inflow
    # Define inflow parameters
    U_h = 8    # streamwise hub height inflow velocity [m/s]
    V_h = 0    # lateral hub height inflow velocity [m/s]
    I0  = 0.10 # hub height total inflow turbulence intensity [-]

    # Preallocate analytical velocity field
    U0 = np.zeros((nx, ny, nz)) # streamwise velocity
    V0 = np.zeros((nx, ny, nz)) # lateral velocity

    # Preallocate streamwise velocity with adiabatic log law inflow
    u_s   = 0.5  # friction velocity [m/s]
    kappa = 0.4  # von Kármán constant [-]
    z0    = 0.15 # roughness length [m]
    
    U_in = np.zeros((nz))
    U_in[1:] = (u_s/kappa) * np.log(z[1:]/z0)
    
    U0[:,:,:] = U_in

    print('Inflow preallocated.')
    
    #%% Run solution
    start_time = time.time()
    flowdata, P = lateralSolution("original", n_t, x_t, y_t, z_t, yaws, x, y, z, U0, U_h, V0, I0, rho=1.225, zh=90.0, D=126.0)
    end_time = time.time()

    #%% Timing
    execution_time = end_time - start_time
    mins = execution_time // 60
    secs = execution_time % 60
    print("Solution execution time: {:.0f}m {:.1f}s".format(mins, secs))
    
    #%% Hub height velocities
    U_zh = flowdata.U.interp(z=zh)
    V_zh = flowdata.V.interp(z=zh)

    #%% Plotting
    fig, ax = plt.subplots(1, 1)
    levels = np.linspace(0,0.6,101)
    p = ax.contourf(x/D,y/D,1-(U_zh.T/U_h),cmap='jet',levels=levels)
    ax.set_xlabel('$x/D$')
    ax.set_ylabel('$y/D$')
    cbar = fig.colorbar(p, ticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    cbar.set_label('$1-(U/U_h)$')

    fig, ax = plt.subplots(1, 1)
    # levels = np.linspace(0,0.32,101)
    p = ax.contourf(x/D,y/D,V_zh.T/U_h,cmap='jet')
    ax.set_xlabel('$x/D$')
    ax.set_ylabel('$y/D$')
    cbar = fig.colorbar(p)
    cbar.set_label('$V_h/U_h$')
    
    # fig, ax = plt.subplots(1,1)
    # for i, i_t in enumerate(list(range(0,n_t,3))):
    #     ax.plot(delta[i_t,:]/D, label=str(i+1 ))
    # ax.set_xlabel('$x/D$')
    # ax.set_ylabel('$\delta_1/D$')
    # ax.legend(title='Row', bbox_to_anchor=(1.05,0.5), loc='center left')