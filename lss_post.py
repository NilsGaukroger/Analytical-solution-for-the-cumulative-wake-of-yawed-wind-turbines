# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 14:43:27 2022

@author: nilsg
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.style.use('default')
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams["legend.scatterpoints"] = 1
plt.rcParams["legend.numpoints"] = 1
plt.rcParams['grid.linestyle'] = ':' # Dotted gridlines
mpl.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.grid']=True
yd = dict(rotation=0,ha='right') # I couldn't find a way to customize these, so use a dict everytime..
plt.close('all')

os.chdir('C:/Users/nilsg/OneDrive/Documents/EWEM/Thesis/NUMERICAL/PyWakeEllipSys/')

from post_utils import windTurbine, windFarm, flowcase, flowcaseGroup
# import post_utils as pu

FigureSize = tuple(x/3 for x in [25.87,11.69])

NREL5MW = windTurbine(D=126, zh=90, CT=1.0, TSR=7.5, wr=True)

lss_wf = windFarm(Uinf=8, ti=0.04, x_wt=[0], y_wt=[0], wts=[NREL5MW], yaws=[40])

data_path = r'C:/Users/nilsg/OneDrive/Documents/EWEM/Thesis/NUMERICAL/PyWakeEllipSys/lss/data/'

#%% Parameters to investigate
yaw_vals = [-40,-30,-20,-10,0,10,20,30,40]
ct_vals  = [0.2, 0.4, 0.6, 0.8]
ti_vals  = [0.04, 0.06, 0.08, 0.1]

#%% Velocity profile: Effect of yaw angle
pyaws = flowcaseGroup(var='yaw', vals=[x for x in yaw_vals if x>=0], path=data_path, wf=lss_wf)

pyaws.plot_VvelocityProfiles(pos=10, xlim=(-5,5), FigureSize=FigureSize)

#%% Velocity profile: Effect of CT
cts  = flowcaseGroup(var='ct', vals=ct_vals, path=data_path, wf=lss_wf)

cts.plot_VvelocityProfiles(pos=10, xlim=(-5,5), FigureSize=FigureSize)

#%% Velocity profile: Effect of T.I.
tis  = flowcaseGroup(var='ti', vals=ti_vals, path=data_path, wf=lss_wf)

tis.plot_VvelocityProfiles(pos=10, xlim=(-5,5), FigureSize=FigureSize)

#%% Velocity profile: Effect of W.R.
# nwr  = flowcaseGroup(var='wr', vals=yaw_vals, path=data_path, wf=lss_wf)

# nwr.plot_VvelocityProfiles(pos=10, xlim=(-5,5), FigureSize=FigureSize)

#%% Velocity profile: Effect of downstream distance
for fc in pyaws.flowcases:
    fc.plot_contourWithProfiles(poss=[5,7,9,11,13], FigureSize=FigureSize)

#%% Self-similarity: Effect of yaw angle
pyaws_no0 = flowcaseGroup(var='yaw', vals=[x for x in yaw_vals if x>0], path=data_path, wf=lss_wf)

pyaws_no0.plot_LSS([5,7,9,11,13], wcm='Gaussian', wwm='Gaussian', xlim=(-4,4), FigureSize=FigureSize)

#%% Self-similarity: Effect of CT
cts.plot_LSS([5,7,9,11,13], wcm='Gaussian', wwm='Gaussian', xlim=(-4,4), FigureSize=FigureSize)

#%% Self-similarity: Effect of T.I.
tis.plot_LSS([5,7,9,11,13], wcm='Gaussian', wwm='Gaussian', xlim=(-4,4), FigureSize=FigureSize)

#%% Self-similarity: Effect of W.R.
# wr.plot_LSS([5,7,9,11,13], wcm='Gaussian', wwm='Gaussian', xlim=(-5,5))

#%% Wake centres and edges
poss = np.linspace(3,29,100)
for fc in pyaws_no0.flowcases:
    fc.plot_wakeCentreAndEdges(poss, wcm='Gaussian', wwm='integral', xlim=(-1,np.max(poss)), ylim=(-3,3), FigureSize=FigureSize)

# pyaws_no0.flowcases[-1].plot_wakeCentreAndEdges(np.linspace(3,20,100), wcm='Gaussian', wwm='integral', FigureSize=FigureSize, xlim=(None,20), ylim=(-2.5,2.5))
    
#%% Self-similarity: Fitted Gaussian
for fc in pyaws_no0.flowcases:
    fc.plot_SS_fitGaussian([5,7,9,11,13], wcm='Gaussian', wwm='Gaussian', xlim=(-4,4), FigureSize=FigureSize)

#%% Debugging

#%% Deleting fig folder
# import shutil
# import os
# import stat

# path = "C:/Users/nilsg/OneDrive/Documents/EWEM/Thesis/NUMERICAL/fig"

# os.chmod(path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
# shutil.rmtree(path, ignore_errors=False)

# print("File deleted")