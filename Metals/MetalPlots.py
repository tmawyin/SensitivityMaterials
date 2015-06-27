import numpy as np
import matplotlib.pyplot as plt

""" ---------- BOXPLOTS ----------  """
""" Will use OutputResults_.txt files """

# Function to generate boxplot based on data
def Bplot(data,label,ylabel,trueVal):
    fig = plt.figure(dpi=600)
    ax = plt.subplot(111)
    bp = plt.boxplot(data, notch=0, sym='o', vert=1, whis=1.5,patch_artist=True)
    plt.setp(bp['boxes'], color='black',linewidth=1.5,facecolor='darkkhaki')
    plt.setp(bp['whiskers'], color='black',linewidth=1.5)
    plt.setp(bp['caps'], color='black',linewidth=1.5)
    plt.setp(bp['medians'], color='darkgreen',linewidth=1.5)
    plt.setp(bp['fliers'], color='grey', marker='o')
    ax.axhline(y=trueVal,xmin=0,xmax=1,c="r",linewidth=2.0,zorder=0,linestyle='--')

    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel,fontsize = 24)
#     ax.set_xlabel(r'Variability',fontsize = 24)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(18)
    ax.set_xticklabels(label,fontsize=18)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
       
    for i in range(len(data)):
        med = bp['medians'][i]
        plt.plot([np.average(med.get_xdata())], [np.average(data[i])],color='r', marker='*', markeredgecolor='k',markersize=10,label="Mean")
    
    fig.tight_layout()
    savingTitle = ylabel.translate(None,'${}')
    fig.savefig(''.join(['Plots/',funcFolder,'/Boxplots/%s.eps'])%(savingTitle),format='eps')


# Loading files
funcFolder = 'F_rho'
folder = ''.join(['Plots/',funcFolder,'/Boxplots/'])
file_name1 = ''.join(['Plots/',funcFolder,'/Boxplots/OutputResults_001pct.txt'])
file_name2 = ''.join(['Plots/',funcFolder,'/Boxplots/OutputResults_03pct.txt'])
file_name3 = ''.join(['Plots/',funcFolder,'/Boxplots/OutputResults_05pct.txt'])
file_name4 = ''.join(['Plots/',funcFolder,'/Boxplots/OutputResults_1pct.txt'])
file_name5 = ''.join(['Plots/',funcFolder,'/Boxplots/OutputResults_25pct.txt'])
file_name6 = ''.join(['Plots/',funcFolder,'/Boxplots/OutputResults_5pct.txt'])
# file_name7 = ''.join(['Plots/',funcFolder,'/Boxplots/OutputResults_10pct.txt'])
   
data1 = np.loadtxt(file_name1)
data2 = np.loadtxt(file_name2)
data3 = np.loadtxt(file_name3)
data4 = np.loadtxt(file_name4)
data5 = np.loadtxt(file_name5)
data6 = np.loadtxt(file_name6)
# data7 = np.loadtxt(file_name7)
   
   
label = ['0.01%','0.3%','0.5%','1%','2.5%','5%','10%']
 
# Cohesive Energy
Ecoh_data = [data1[:,1],data2[:,1],data3[:,1],data4[:,1],data5[:,1],data6[:,1]]#,data7[:,1]]
Bplot(Ecoh_data,label,r"$E_{cohesive}$ $[eV]$",-3.36)
  
# Lattice Parameter
Latt_data = [data1[:,2],data2[:,2],data3[:,2],data4[:,2],data5[:,2],data6[:,2]]#,data7[:,2]]
Bplot(Latt_data,label,r"$Lattice$ $[A]$",4.05)
  
# Grain Boundary Energy [mJ/m2]
GB_data = [data1[:,3],data2[:,3],data3[:,3],data4[:,3],data5[:,3],data6[:,3]]#,data7[:,3]]
Bplot(GB_data,label,r"$E_{GrainBoundary}$ $[mJm^{-2}]$",494.0)
  
# Vacancy Formation Energy [eV]
Vac_data = [data1[:,4],data2[:,4],data3[:,4],data4[:,4],data5[:,4],data6[:,4]]#,data7[:,4]]
Bplot(Vac_data,label,r"$E_{Vacancy}$ $[eV]$",0.675)
  
# Stacking Fault Energy [mJ/m2]
Stack_data = [data1[:,5],data2[:,5],data3[:,5],data4[:,5],data5[:,5],data6[:,5]]#,data7[:,5]]
Bplot(Stack_data,label,r"$E_{Stacking}$ $[mJm^{-2}]$",145.2)
  
# Interstitial Energy [eV]
Inter_data = [data1[:,6],data2[:,6],data3[:,6],data4[:,6],data5[:,6],data6[:,6]]#,data7[:,6]]
Bplot(Inter_data,label,r"$E_{Interstitial}$ $[eV]$",2.789)
  
# Surface Energy [mJ/m2]
Surf_data = [data1[:,7],data2[:,7],data3[:,7],data4[:,7],data5[:,7],data6[:,7]]#,data7[:,7]]
Bplot(Surf_data,label,r"$E_{Surface}$ $[mJm^{-2}]$",943.6)
  
# C11 [GPa]
C11_data = [data1[:,8],data2[:,8],data3[:,8],data4[:,8],data5[:,8],data6[:,8]]#,data7[:,8]]
Bplot(C11_data,label,r"$C_{11}$ $[GPa]$",113.8)
  
# C12 [GPa]
C12_data = [data1[:,9],data2[:,9],data3[:,9],data4[:,9],data5[:,9],data6[:,9]]#,data7[:,9]]
Bplot(C12_data,label,r"$C_{12}$ $[GPa]$",61.55)
 
# C44 [GPa]
C44_data = [data1[:,10],data2[:,10],data3[:,10],data4[:,10],data5[:,10],data6[:,10]]#,data7[:,10]]
Bplot(C44_data,label,r"$C_{44}$ $[GPa]$",31.59)

# plt.show()

'''
""" ---------- SCVR PLOTS ----------  """
""" From SCVR data from loo.py """
def scvrPlot(data,title,saveName):
    a = np.arange(0,len(data),1)
    fig = plt.figure(dpi=600)
    ax = fig.add_subplot(111)
    ax.plot(a,data,'o',ms=6.0)
    ax.axhline(y=3,xmin=0,xmax=1,c="r",linewidth=2.0,zorder=0,linestyle='--')
    ax.axhline(y=-3,xmin=0,xmax=1,c="r",linewidth=2.0,zorder=0,linestyle='--')
    ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
#     ax.set_title(title)
    ax.set_xlabel(r'Sample Points',fontsize=16)
    ax.set_ylabel(r'SCVR',fontsize=16)
    ax.set_xlim(0,len(a))
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    ax.set_xticklabels('')
    # ax.set_ylim(-5,5)
#     fig.text(0.5, 0.04, 'Sample Points', ha='center', va='center')
#     fig.text(0.02, 0.5, 'Standardized Cross Validation Residuals', ha='center', va='center', rotation='vertical') 
    fig.tight_layout()
    savingTitle = saveName.translate(None,'${}')
    fig.savefig('Plots/F_rho/SCVR/%s.eps'%(savingTitle),format='eps')

# Loading files
pct = '5pct'
file_name = 'Plots/F_rho/SCVR/scvr_Stacking.dat'
observable = 'Estack'

title = r"$C_{11}$ $[eV]$"
saveName = ''.join([observable,'_',pct])
 
data = np.genfromtxt(file_name,delimiter='\t',filling_values=np.NaN)
scvrPlot(data,title,saveName)
 
plt.show()

'''