''' Generates Histograms based on optimization code '''

import numpy as np
import matplotlib.pyplot as plt
ErrorF = np.loadtxt('Error_10pct_500pts')
print np.mean(ErrorF)

fig = plt.figure(dpi=600)
ax = fig.add_subplot(111)
n, bins, patches = ax.hist(ErrorF, 80, normed=1, facecolor='green', alpha=0.75)
ax.axvline(x=np.mean(ErrorF),ymin=0,ymax=1,c="r",linewidth=2.0,zorder=0,linestyle='--')
ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
ax.set_xlabel('Error',fontsize=18)
ax.set_ylabel('Frequency',fontsize=18)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(16)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(16)

fig.tight_layout()
fig.savefig('PotentialHistogram_10pct_.eps',format='eps')
# np.savetxt('Error',ErrorF.reshape(1,nTrials))
# plt.show()
