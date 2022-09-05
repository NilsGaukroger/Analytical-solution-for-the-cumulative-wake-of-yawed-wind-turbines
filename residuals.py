# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 14:27:53 2022

@author: nilsg
"""

import pandas as pd
import matplotlib.pyplot as plt
from post_utils2 import set_size

textwidth = 448.0 # [pt]

# Edit the font, font size, and axes width
plt.rcParams.update({
  "text.usetex": True,
  "font.family": "Tex Gyre Adventor"
})
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['lines.linewidth'] = 2

fig_path = 'C:/Users/nilsg/Dropbox/Apps/Overleaf/Thesis/figures/02_cfdSetup/'

names = ['U','V','W','P','k','eps']
res = pd.read_csv('grid.res', header=0, names = ['U','V','W','P','k','eps'], index_col=0, skiprows=[0,1,2,3,4,5,6,7,8,198,199,200,201,202,203], delim_whitespace=True)

fig, ax = plt.subplots(1,1,figsize=set_size(width=textwidth,fraction=0.9))

ax.plot(res[names[:]],linewidth=2,label=['$U$','$V$','$W$','$P$','$k$',r'$\varepsilon$'])
ax.legend(bbox_to_anchor=(1.025,0.5),loc='center left',frameon=False,title='Variable')
ax.set_xlim(left=0, right = len(res))
ax.set_ylim(top=0,bottom=-12)
ax.set_xlabel('Iterations')
ax.set_ylabel('$\log_{10}(\\rho)$')
ax.axvline(190,c='k',ls='--')
ax.text(190/2,-1,'Coarser mesh',horizontalalignment='center')
ax.text(190+((len(res)-190)/2),-1,'Finer mesh',horizontalalignment='center')

# Save figure
fig.savefig(fig_path + 'res.pdf', bbox_inches='tight')
plt.show()