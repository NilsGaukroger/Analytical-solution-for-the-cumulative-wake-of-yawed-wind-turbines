# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 16:04:57 2022

@author: nilsg
"""

import numpy as np
import matplotlib.pyplot as plt

def MOA(x_Ds, data, cDs, figfile, r=2, plot_x_Ds=[]):
    # Setup
    n_x  = len(x_Ds) # number of locations
    n_cD = len(cDs)  # number of grid levels
    
    # Preallocation
    RE         = np.zeros((n_x)) # Richardson extrapolated value
    # xStore     = np.zeros((n_x, n_cD+1)) # g values
    errorStore = np.zeros((n_cD, n_x)) # errors
    
    #%% Coefficient matrix (h=1) for n_cD sizes with refinement factor 2
    A = np.zeros((n_cD, n_cD))
    for i in range(n_cD):
        for j in range(n_cD):
            A[i,j] = 2**(i*j)
    
    #%% Error functions
    # Absolute error function
    def f(x,g,n):
        return abs(g)*x**n
    
    # Total absolute error function
    def f_total(x,gvec):
        f = 0
        for i in range(len(gvec)):
            f += gvec[i]*x**(i+1)
        return abs(f)
    
    #%% Write outputs
    for i in range(n_x):
        # Right hand side b
        b = data[i, ::-1] # data reversed
        
        # Solve Ax = b
        x = np.linalg.solve(A, b)
        
        RE[i] = x[0]
        
        # Store errors
        # xStore[i,0]       = i
        # xStore[i,1:]      = np.transpose(x)
        errorStore[:,i]   = b - RE[i]
        
        if x_Ds[i] in plot_x_Ds:
            # Plot absolute errors
            fig, ax = plt.subplots(1,1)
            
            # Create logspace in x
            xlog = np.logspace(-1, 2, 100, endpoint=True)
            
            # Set x-axis limits
            ax.set_xlim(10**(-1), 10**(2))
            
            # Log-log plot errors
            ax.loglog(xlog,
                      f_total(xlog, x[1:]),
                      label='Total error')
            ax.loglog(xlog,
                      f(xlog, x[1], 1),
                      label='1st order')
            ax.loglog(xlog,
                      f(xlog, x[2], 1),
                      label='2nd order')
            # ax.loglog(xlog,
            #           f(xlog, x[3], 1),
            #           label='3rd order')
            
            # Something else
            ax.scatter(A[1,:],
                        np.abs(errorStore[:,i]),
                        label='$\\textrm{D}_\\textrm{n}$ - RE')
            
            # Set axes labels
            ax.set_xlabel('Normalised grid size, $h$')
            ax.set_ylabel('Absolute error')
            
            # Create legend
            ax.legend(loc = 4, scatterpoints = 1)
            
            # Add title with downstream distance
            fig.suptitle('$x/D = {:.1f}$'.format(x_Ds[i]))
            
            # Save figure 
            fig.savefig(figfile + 'x_D_' + str(x_Ds[i]) + '.pdf')
            
            # Show figure
            plt.show()
    
        # put errorStore back in correct order
        errorStore[:,i] = errorStore[::-1,i]
    
    return RE, errorStore