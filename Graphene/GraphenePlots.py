#! Program to generate plots and figures of given data

import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd


# Set ups latex formatting
#plt.rc('text', usetex = True)
#plt.rc('font', family = 'normal')

''' --- FUNCTIONS --- '''
# Trims data - Eg. trimData(data4[:,1],maxVal,minVal,True)
def trimData(data,maxVal,minVal,pltData):
    data = data.reshape(len(data),1)
    data_trim = data[np.logical_and(data[:,0] < maxVal, data[:,0] > minVal)]
    if pltData:
        plt.figure()
        plt.scatter(range(len(data_trim)),data_trim[:,0])
        plt.grid()
    return data_trim

# Removes NaN values from any vector
def delData(data):
    data = data.reshape(len(data),1)
    data_del = data[~np.isnan(data)]
    return data_del

""" LJ SCATTER PLOTS """
file_name = 'Plots/LJ/results_LJ_full.txt'
data = np.loadtxt(file_name)

# fig = plt.figure(1,figsize=(14,8),dpi=600)
fig = plt.figure(1,dpi=600) #figsize=(3.375,3.375),
# fig = plt.figure()
ax1 = plt.subplot(2,1,1)
ax1.axhline(y=1150,xmin=0,xmax=500,c="r",linewidth=1,zorder=0,linestyle='--')
plt.scatter(data[:,0], data[:,1], c = "b")
plt.xlim(0,500)
# ax1.set_xlabel("(a)")# $C_{11}$ elastic constant")
ax1.set_ylabel("$C_{11}$ $[GPa]$",fontsize=14)
ax1.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
plt.setp(ax1.get_xticklabels(), visible=False)
for tick in ax1.yaxis.get_major_ticks():
    tick.label.set_fontsize(12) 

ax2 = plt.subplot(2,1,2)
plt.scatter(data[:,0], data[:,2], c = "b")
plt.xlim(0,500)
ax2.set_xlabel("Number of samples",fontsize=14)# $C_{12}$ elastic constant")
ax2.set_ylabel("$C_{12}$ $[GPa]$",fontsize=14)
ax2.axhline(y=150,xmin=0,xmax=500,c="r",linewidth=1.0,zorder=0,linestyle='--')
ax2.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
ax2.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)

plt.setp(ax2.get_xticklabels(), visible=False)
for tick in ax2.yaxis.get_major_ticks():
    tick.label.set_fontsize(12) 

# ax3 = plt.subplot(2,2,3)
# plt.scatter(data[:,0], data[:,3], c = "b")
# plt.xlim(0,500)
# plt.ylim(-7.495,-7.460)
# # ax3.set_xlabel("Realizations \n (c)")# Cohesive energy")
# ax3.set_ylabel("$E_{cohesive}$ $[eV]$",fontsize=16)
# ax3.axhline(y=-7.48,xmin=0,xmax=500,c="r",linewidth=3.0,zorder=0,linestyle='--')
# ax3.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
# ax3.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)

# ax4 = plt.subplot(2,2,4)
# plt.scatter(data[:,0], data[:,4], c = "b")
# plt.xlim(0,500)
# plt.ylim(7.152,7.166)
# # ax4.set_xlabel("Realizations \n (d)")# Vacancy formation energy")
# ax4.set_ylabel("$E_{vacancy}$ $[eV]$",fontsize=16)
# ax4.axhline(y=7.157,xmin=0,xmax=500,c="r",linewidth=3.0,zorder=0,linestyle='--')
# ax4.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
# ax4.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)

fig.tight_layout()
plt.savefig('Plots/LJ/LJ_paper.eps',format='eps')
plt.show()


""" ---------- BOXPLOTS ----------  """  
# def Bplot(data,label,ylabel,logCond,trueVal):
#     fig = plt.figure(dpi=600)
#     ax = plt.subplot(111)
#     bp = plt.boxplot(data, notch=0, sym='o', vert=1, whis=1.5,patch_artist=True)
#     plt.setp(bp['boxes'], color='black',linewidth=1.5,facecolor='darkkhaki')
#     plt.setp(bp['whiskers'], color='black',linewidth=1.5)
#     plt.setp(bp['caps'], color='black',linewidth=1.5)
#     plt.setp(bp['medians'], color='darkgreen',linewidth=1.5)
#     plt.setp(bp['fliers'], color='grey', marker='o')
#     # ax.axhline(y=trueVal,xmin=0,xmax=1,c="r",linewidth=2.0,zorder=0,linestyle='--')
      
#     ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
#     ax.set_axisbelow(True)
#     ax.set_ylabel(ylabel,fontsize=24)
#     ax.set_xlabel(r'Variability',fontsize=18)
#     ax.set_xticklabels(label,fontsize=18)
#     for tick in ax.yaxis.get_major_ticks():
#         tick.label.set_fontsize(18)
#     ax.get_xaxis().tick_bottom()
#     ax.get_yaxis().tick_left()
#     if logCond:
#         ax.set_yscale('log')
         
#     for i in range(len(data)):
#         med = bp['medians'][i]
#         plt.plot([np.average(med.get_xdata())], [np.average(data[i])],color='r', marker='*', markeredgecolor='k',markersize=10,label="Mean")
      
#     fig.tight_layout()
#     savingTitle = ylabel.translate(None,'${}')
#     fig.savefig('Plots/Boxplots/%s.eps'%(savingTitle),format='eps')
 
# # Loading files
# file_name1 = 'Plots/Boxplots/CompleteResults_001pct.txt'
# file_name2 = 'Plots/Boxplots/CompleteResults_03pct.txt'
# file_name3 = 'Plots/Boxplots/CompleteResults_05pct.txt'
# file_name4 = 'Plots/Boxplots/CompleteResults_1pct.txt'
# file_name5 = 'Plots/Boxplots/CompleteResults_25pct.txt'
# file_name6 = 'Plots/Boxplots/CompleteResults_5pct.txt'
     
# data1 = np.genfromtxt(file_name1,delimiter='\t',filling_values=np.NaN)
# data2 = np.genfromtxt(file_name2,delimiter='\t',filling_values=np.NaN)
# data3 = np.genfromtxt(file_name3,delimiter='\t',filling_values=np.NaN)
# data4 = np.genfromtxt(file_name4,delimiter='\t',filling_values=np.NaN)
# data5 = np.genfromtxt(file_name5,delimiter='\t',filling_values=np.NaN)
# data6 = np.genfromtxt(file_name6,delimiter='\t',filling_values=np.NaN)
  
# label = ['0.01%','0.3%','0.5%','1%','2.5%','5%']
  
# # C11
# C11_data = [delData(data1[:,1]),delData(data2[:,1]),delData(data3[:,1]),delData(data4[:,1]),delData(data5[:,1]),delData(data6[:,1])]
# Bplot(C11_data,label,r"$C_{11}$ $[GPa]$",True,1150)
      
# # C12
# C12_data = [delData(data1[:,2]),delData(data2[:,2]),delData(data3[:,2]),delData(data4[:,2]),delData(data5[:,2]),delData(data6[:,2])]
# Bplot(C12_data,label,r"$C_{12}$ $[GPa]$",True,150)
      
# # Cohesive Energy [eV]
# Ecoh_data = [delData(data1[:,3]),delData(data2[:,3]),delData(data3[:,3]),delData(data4[:,3]),delData(data5[:,3]),delData(data6[:,3])]
# Bplot(Ecoh_data,label,r"$E_{cohesive}$ $[eV]$",False,-7.49)
       
# # Vacancy Formation Energy [eV]
# maxVal = 150
# minVal = -150
# vac1 = trimData(delData(data4[:,4]),maxVal,minVal,False)
# vac25 = trimData(delData(data5[:,4]),maxVal,minVal,False)
# vac5 = trimData(delData(data6[:,4]),maxVal,minVal,False)
# Vac_data = [delData(data1[:,4]),delData(data2[:,4]),delData(data3[:,4]),vac1,vac25,vac5]
# Bplot(Vac_data,label,r"$E_{Vacancy}$ $[eV]$",False,7.5)
       
# # Bond Rotation Energy [eV]
# maxVal = 5000
# minVal = -15000
# rot1 = trimData(delData(data4[:,5]),maxVal,minVal,False)
# rot25 = trimData(delData(data5[:,5]),maxVal,minVal,False)
# rot5 = trimData(delData(data6[:,5]),maxVal,minVal,False)
# Rot_data = [delData(data1[:,5]),delData(data2[:,5]),delData(data3[:,5]),rot1,rot25,rot5]
# Bplot(Rot_data,label,r"$E_{bond-rotation}$ $[eV]$",False,5.0)
       
# # Divacancy Energy [eV]
# maxVal = 150
# minVal = -150
# dv1 = trimData(delData(data4[:,6]),maxVal,minVal,False)
# dv25 = trimData(delData(data5[:,6]),maxVal,minVal,False)
# dv5 = trimData(delData(data6[:,6]),maxVal,minVal,False)
# Divac_data = [delData(data1[:,6]),delData(data2[:,6]),delData(data3[:,6]),dv1,dv25,dv5]
# Bplot(Divac_data,label,r"$E_{divacancy}$ $[eV]$",False,7.7)
       
# # Grain Boundary Energy [eV]
# maxVal = 10
# minVal = -7.5
# gb1 = trimData(delData(data4[:,7]),maxVal,minVal,False)
# gb25 = trimData(delData(data5[:,7]),maxVal,minVal,False)
# gb5 = trimData(delData(data6[:,7]),maxVal,minVal,False)
# Gb_data = [delData(data1[:,7]),delData(data2[:,7]),delData(data3[:,7]),gb1,gb25,gb5]
# Bplot(Gb_data,label,r"$E_{GrainBoundary}$ $[eV]$",False,0.223)
       
# # Thermal Conductiviy [W/mK]
# Tcond_data = [delData(data1[:,8]),delData(data2[:,8]),delData(data3[:,8]),delData(data4[:,8]),delData(data5[:,8]),delData(data6[:,8])]
# Bplot(Tcond_data,label,r"$T_{conductivity}$ $[Wm^{-1}K^{-1}]$",False,450)
   
# # plt.show()


""" ---------- PIE CHARTS - STOCHASTIC MODEL ----------  """
# label = np.char.array([r'$Q_{CC}$',r'$\alpha_{CC}$',r'$\beta_{CC1}$',r'$\beta_{CC2}$',r'$\beta_{CC3}$',
#                        r'$rc^{LJmin}_{CC}$',r'$rc^{LJmax}_{CC}$',r'$\epsilon_{CC}$',r'$\sigma_{CC}$',r'$\epsilon^{T}_{CCCC}$'])
#     
# colors = ['lightcoral','red','gold','lightskyblue','white','yellowgreen','blue',
#           'pink', 'darkgreen','yellow','grey','violet','magenta','cyan'] 
#     
# # --- C11 Elastic Constant ---
# file_name = 'Pie/C11_pie_5.txt'
# p = np.loadtxt(file_name)
# pct = 100.*p
# explode = [0.05]*len(p)
# 
# fig = plt.figure(1)
# ax1 = fig.add_axes([0.25, 0.01, 0.8, 0.95])
# patches, texts = ax1.pie(p, colors=colors, startangle=90, radius=1, explode=explode)
# labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(label, pct)]
# legd = ax1.legend(patches, labels, loc='center', bbox_to_anchor=(-0.13, 0.5), fontsize=11,borderaxespad=0.)
# #fig.savefig('Pie/C11_pie_5.eps', format='eps', bbox_extra_artist=(legd), bbox_inches='tight')
#    
# # --- C12 Elastic Constant ---
# file_name = 'Pie/C12_pie_5.txt'
# p = np.loadtxt(file_name)
# pct = 100.*p
# explode = [0.05]*len(p)
#    
# fig = plt.figure(2)
# ax1 = fig.add_axes([0.25, 0.01, 0.8, 0.95])
# patches, texts = ax1.pie(p, colors=colors, startangle=90, radius=1, explode=explode)
# labels = ['{0} - {1:1.2f} \%'.format(i,j) for i,j in zip(label, pct)]
# legd = ax1.legend(patches, labels, loc='center', bbox_to_anchor=(-0.13, 0.5), fontsize=11,borderaxespad=0.)
# #fig.savefig('Pie/C12_pie_5.eps', format='eps', bbox_extra_artist=(legd), bbox_inches='tight')
#    
# # --- Cohesive Energy ---
# file_name = 'Pie/Ecoh_pie_5.txt'
# p = np.loadtxt(file_name)
# pct = 100.*p
# explode = [0.05]*len(p)
#    
# fig = plt.figure(3)
# ax1 = fig.add_axes([0.25, 0.01, 0.8, 0.95])
# patches, texts = ax1.pie(p, colors=colors, startangle=90, radius=1, explode=explode)
# labels = ['{0} - {1:1.2f} \%'.format(i,j) for i,j in zip(label, pct)]
# legd = ax1.legend(patches, labels, loc='center', bbox_to_anchor=(-0.13, 0.5), fontsize=11,borderaxespad=0.)
# #fig.savefig('Pie/Eco_pie_5.eps', format='eps', bbox_extra_artist=(legd), bbox_inches='tight')
#   
# # --- Vacancy Formation Energy ---
# file_name = 'Pie/Evac_pie_5.txt'
# p = np.loadtxt(file_name)
# pct = 100.*p
# explode = [0.05]*len(p)
#    
# fig = plt.figure(4)
# ax1 = fig.add_axes([0.25, 0.01, 0.8, 0.95])
# patches, texts = ax1.pie(p, colors=colors, startangle=90, radius=1, explode=explode)
# labels = ['{0} - {1:1.2f} \%'.format(i,j) for i,j in zip(label, pct)]
# legd = ax1.legend(patches, labels, loc='center', bbox_to_anchor=(-0.13, 0.5), fontsize=11,borderaxespad=0.)
# #fig.savefig('Pie/Eva_pie_5.eps', format='eps', bbox_extra_artist=(legd), bbox_inches='tight')
#    
# plt.show()


""" ---------- MAIN EFFECTS - STOCHASTIC MODEL ----------  """
# label = np.char.array([r'$Q_{CC}$',r'$\alpha_{CC}$',r'$\beta_{CC1}$',r'$\beta_{CC2}$',r'$\beta_{CC3}$',
#                        r'$rc^{LJmin}_{CC}$',r'$rc^{LJmax}_{CC}$',r'$\epsilon_{CC}$',r'$\sigma_{CC}$',r'$\epsilon^{T}_{CCCC}$'])
   
# file_name = 'BackUp_Data/MainEffect/1pct/Evac_me_1.txt'
# p = np.loadtxt(file_name)   # p has 21 columns = X, 10 for Y, 10 for Error
   
# fig = plt.figure(1)
# ax1 = fig.add_subplot(521)
# ax1.plot(p[:,0],p[:,1],'b',linewidth=2)
# ax1.plot(p[:,0],p[:,1]+p[:,11],'--r')
# ax1.plot(p[:,0],p[:,1]-p[:,11],'--r')
# ax1.set_xlabel(label[0])
# plt.locator_params(nbins=5)
   
# ax2 = fig.add_subplot(522)
# ax2.plot(p[:,0],p[:,2],'b',linewidth=2)
# ax2.plot(p[:,0],p[:,2]+p[:,12],'--r')
# ax2.plot(p[:,0],p[:,2]-p[:,12],'--r')
# ax2.set_xlabel(label[1])
# plt.locator_params(nbins=5)
   
# ax3 = fig.add_subplot(523)
# ax3.plot(p[:,0],p[:,3],'b',linewidth=2)
# ax3.plot(p[:,0],p[:,3]+p[:,13],'--r')
# ax3.plot(p[:,0],p[:,3]-p[:,13],'--r')
# ax3.set_xlabel(label[2])
# plt.locator_params(nbins=5)
   
# ax4 = fig.add_subplot(524)
# ax4.plot(p[:,0],p[:,4],'b',linewidth=2)
# ax4.plot(p[:,0],p[:,4]+p[:,14],'--r')
# ax4.plot(p[:,0],p[:,4]-p[:,14],'--r')
# ax4.set_xlabel(label[3])
# plt.locator_params(nbins=5)
   
# ax5 = fig.add_subplot(525)
# ax5.plot(p[:,0],p[:,5],'b',linewidth=2)
# ax5.plot(p[:,0],p[:,5]+p[:,15],'--r')
# ax5.plot(p[:,0],p[:,5]-p[:,15],'--r')
# ax5.set_xlabel(label[4])
# plt.locator_params(nbins=5)
    
# ax6 = fig.add_subplot(526)
# ax6.plot(p[:,0],p[:,6],'b',linewidth=2)
# ax6.plot(p[:,0],p[:,6]+p[:,16],'--r')
# ax6.plot(p[:,0],p[:,6]-p[:,16],'--r')
# ax6.set_xlabel(label[5])
# plt.locator_params(nbins=5)
   
# ax7 = fig.add_subplot(527)
# ax7.plot(p[:,0],p[:,7],'b',linewidth=2)
# ax7.plot(p[:,0],p[:,7]+p[:,17],'--r')
# ax7.plot(p[:,0],p[:,7]-p[:,17],'--r')
# ax7.set_xlabel(label[6])
# plt.locator_params(nbins=5)
   
# ax8 = fig.add_subplot(528)
# ax8.plot(p[:,0],p[:,8],'b',linewidth=2)
# ax8.plot(p[:,0],p[:,8]+p[:,18],'--r')
# ax8.plot(p[:,0],p[:,8]-p[:,18],'--r')
# ax8.set_xlabel(label[7])
# plt.locator_params(nbins=5)
   
# ax9 = fig.add_subplot(529)
# ax9.plot(p[:,0],p[:,9],'b',linewidth=2)
# ax9.plot(p[:,0],p[:,9]+p[:,19],'--r')
# ax9.plot(p[:,0],p[:,9]-p[:,19],'--r')
# ax9.set_xlabel(label[8])
# plt.locator_params(nbins=5)
   
# ax10 = fig.add_subplot(5,2,10)
# ax10.plot(p[:,0],p[:,10],'b',linewidth=2)
# ax10.plot(p[:,0],p[:,10]+p[:,20],'--r')
# ax10.plot(p[:,0],p[:,10]-p[:,20],'--r')
# ax10.set_xlabel(label[9])
# plt.locator_params(nbins=5)
   
# # fig.suptitle(r'$E_{vacancy}$ at $1\%$ Variability',fontsize=11)
# fig.tight_layout()
# fig.subplots_adjust(top=0.92,hspace=0.5)
# plt.setp([a.get_xticklabels() for a in fig.axes[:-2]], visible=False)
   
# fig.savefig('BackUp_Data/MainEffect/1pct/Evac_me_1_new.eps', format='eps')
# plt.show()


""" ---------- LEAVE ONE OUT - STOCHASTIC MODEL ----------  """
# def looPlot(data,saveName):
#     fig = plt.figure(dpi=600)
#     ax = fig.add_subplot(111)
#     ax.scatter(data[:,0],data[:,1])
#     ax.plot([minVal,maxVal],[minVal,maxVal], 'r', linewidth=2.0)
#     ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
#     ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
#     ax.get_xaxis().tick_bottom()
#     ax.get_yaxis().tick_left()
# #     ax.set_title(r'Leave-one-out')
# #     fig.text(0.5, 0.04, 'Actual Values', ha='center', va='center')
# #     fig.text(0.02, 0.5, 'Predicted Values', ha='center', va='center', rotation='vertical')   
#     ax.set_xlabel(r'Actual',fontsize=18)
#     ax.set_ylabel(r'Predicted',fontsize=18)
#     for tick in ax.yaxis.get_major_ticks():
#         tick.label.set_fontsize(16)
#     for tick in ax.xaxis.get_major_ticks():
#         tick.label.set_fontsize(16)
#     fig.tight_layout()
#     savingTitle = saveName.translate(None,'${}')
#     fig.savefig('Plots/Loo/1pct/%s.eps'%(savingTitle),format='eps')
#    
# # Loading files
# pct = '1pct'
# file_name = 'Plots/Loo/1pct/loo_EGB_1pct.dat'
# title = r"$E_{grainboundary}$ $[eV]$"
# maxVal = 0.40
# minVal = 0.15
# saveName = ''.join([title,'_',pct])
#  
# # data = np.loadtxt(file_name)
# data = np.genfromtxt(file_name,delimiter=' ',filling_values=np.NaN)
# for i in range(len(data)):
#     if np.mod(i,2)==0 :
#         data[i,1] = data[i,0]
#  
# looPlot(data,saveName)
# plt.show()
# -----------------------------

# file_name1 = 'MainEffect/C11_ap_1.txt'
# file_name2 = 'MainEffect/C12_ap_1.txt'
# file_name3 = 'MainEffect/Eco_ap_1.txt'
# file_name4 = 'MainEffect/Eva_ap_1.txt'
#     
# data1 = np.loadtxt(file_name1)
# data2 = np.loadtxt(file_name2)
# data3 = np.loadtxt(file_name3)
# data4 = np.loadtxt(file_name4)
#     
# fig = plt.figure(1)
#    
# ax1 = fig.add_subplot(221)
# ax1.scatter(data1[:,0],data1[:,1])
# ax1.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
# ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
# ax1.get_xaxis().tick_bottom()
# ax1.get_yaxis().tick_left()
# ax1.set_title(r'$C_{11}$')
# ax1.set_xlabel(r'(a)')
#     
# ax2 = fig.add_subplot(222)
# ax2.scatter(data2[:,0],data2[:,1])
# ax2.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
# ax2.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
# ax2.get_xaxis().tick_bottom()
# ax2.get_yaxis().tick_left()
# ax2.set_title(r'$C_{12}$')
# ax2.set_xlabel(r'(b)')
#     
# ax3 = fig.add_subplot(223)
# ax3.scatter(data3[:,0],data3[:,1])
# ax3.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
# ax3.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
# ax3.get_xaxis().tick_bottom()
# ax3.get_yaxis().tick_left()
# ax3.set_title(r'$E_{cohesive}$')
# ax3.set_xlabel(r'(C)')
#     
# ax4 = fig.add_subplot(224)
# ax4.scatter(data4[:,0],data4[:,1])
# ax4.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
# ax4.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
# ax4.get_xaxis().tick_bottom()
# ax4.get_yaxis().tick_left()
# ax4.set_title(r'$E_{vacancy}$')
# ax4.set_xlabel(r'(d)')
#     
# fig.text(0.5, 0.04, 'Actual Values', ha='center', va='center')
# fig.text(0.02, 0.5, 'Predicted Values', ha='center', va='center', rotation='vertical')
#    
# fig.tight_layout()
# fig.savefig('MainEffect/loo_1.eps', format='eps')
# plt.show()


""" ---------- SCVR - STOCHASTIC MODEL ----------  """
# def scvrPlot(data,title,saveName):
#     a = np.arange(0,len(data),1)
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.scatter(a,data)
#     ax.axhline(y=3,xmin=0,xmax=1,c="r",linewidth=1,zorder=0,linestyle='--')
#     ax.axhline(y=-3,xmin=0,xmax=1,c="r",linewidth=1,zorder=0,linestyle='--')
#     ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
#     ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
#     ax.get_xaxis().tick_bottom()
#     ax.get_yaxis().tick_left()
# #     ax.set_title(title)
#     ax.set_xlabel(r'Sample Points')
#     ax.set_ylabel(r'SCVR')
#     ax.set_xlim(0,len(a))
#     ax.set_xticklabels('')
#     ax.set_ylim(-5,5)
# #     fig.text(0.5, 0.04, 'Sample Points', ha='center', va='center')
# #     fig.text(0.02, 0.5, 'Standardized Cross Validation Residuals', ha='center', va='center', rotation='vertical') 
#     fig.tight_layout()
#     savingTitle = saveName.translate(None,'${}')
#     fig.savefig('Plots/SCVR/%s.eps'%(savingTitle),format='eps')
# 
# # Loading files
# pct = '1pct'
# file_name = 'Plots/SCVR/scvr_E59.dat'
# title = r"$E_{59}$ $[eV]$"
# saveName = ''.join([title,'_',pct])
#  
# data = np.genfromtxt(file_name,delimiter='\t',filling_values=np.NaN)
# scvrPlot(data,title,saveName)
#  
# plt.show()
# -----------------------------
    
# file_name1 = 'MainEffect/C11_scvr_5.txt'
# file_name2 = 'MainEffect/C12_scvr_5.txt'
# file_name3 = 'MainEffect/Ecoh_scvr_5.txt'
# file_name4 = 'MainEffect/Evac_scvr_5.txt'
#    
# data1 = np.loadtxt(file_name1)
# data2 = np.loadtxt(file_name2)
# data3 = np.loadtxt(file_name3)
# data4 = np.loadtxt(file_name4)
#   
# a = np.arange(0,len(data1),1)
#   
# fig = plt.figure(1)
# ax1 = fig.add_subplot(221)
# ax1.scatter(a,data1)
# ax1.axhline(y=3,xmin=0,xmax=1,c="r",linewidth=1,zorder=0,linestyle='--')
# ax1.axhline(y=-3,xmin=0,xmax=1,c="r",linewidth=1,zorder=0,linestyle='--')
# ax1.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
# ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
# ax1.get_xaxis().tick_bottom()
# ax1.get_yaxis().tick_left()
# ax1.set_title(r'$C_{11}$')
# ax1.set_xlabel(r'(a)')
# ax1.set_xlim(0,len(a))
# ax1.set_xticklabels('')
# #ax1.set_ylim(-5,5)
# 
# ax2 = fig.add_subplot(222)
# ax2.scatter(a,data2)
# ax2.axhline(y=3,xmin=0,xmax=1,c="r",linewidth=1,zorder=0,linestyle='--')
# ax2.axhline(y=-3,xmin=0,xmax=1,c="r",linewidth=1,zorder=0,linestyle='--')
# ax2.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
# ax2.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
# ax2.get_xaxis().tick_bottom()
# ax2.get_yaxis().tick_left()
# ax2.set_title(r'$C_{12}$')
# ax2.set_xlabel(r'(b)')
# ax2.set_xticklabels('')
# ax2.set_xlim(0,len(a))
# #ax2.set_ylim(-5,5)
# 
# ax3 = fig.add_subplot(223)
# ax3.scatter(a,data3)
# ax3.axhline(y=3,xmin=0,xmax=1,c="r",linewidth=1,zorder=0,linestyle='--')
# ax3.axhline(y=-3,xmin=0,xmax=1,c="r",linewidth=1,zorder=0,linestyle='--')
# ax3.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
# ax3.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
# ax3.get_xaxis().tick_bottom()
# ax3.get_yaxis().tick_left()
# ax3.set_title(r'$E_{cohesive}$')
# ax3.set_xlabel(r'(c)')
# ax3.set_xlim(0,len(a))
# #ax3.set_ylim(-5,5)
# 
# ax4 = fig.add_subplot(224)
# ax4.scatter(a,data4)
# ax4.axhline(y=3,xmin=0,xmax=1,c="r",linewidth=1,zorder=0,linestyle='--')
# ax4.axhline(y=-3,xmin=0,xmax=1,c="r",linewidth=1,zorder=0,linestyle='--')
# ax4.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
# ax4.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
# ax4.get_xaxis().tick_bottom()
# ax4.get_yaxis().tick_left()
# ax4.set_title(r'$E_{vacancy}$')
# ax4.set_xlabel(r'(d)')
# ax4.set_xlim(0,len(a))
# #ax4.set_ylim(-5,5)
#  
# fig.text(0.5, 0.04, 'Sample Size', ha='center', va='center')
# fig.text(0.02, 0.5, 'Standardized Cross Validation Residuals', ha='center', va='center', rotation='vertical') 
# fig.tight_layout()
# fig.savefig('MainEffect/scvr_5.eps', format='eps')
# plt.show()


""" ---------- SPARSE GRID ----------  """
# from matplotlib.ticker import MaxNLocator
# def sparsePlot(ylabel):
#     fig = plt.figure(dpi=600)
#   
#     ax1 = fig.add_subplot(121)
#     ax1.plot(data[:,0],data[:,1],'-o',c='b',lw=3.0,ms=8.0)
#     ax1.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
#     ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
#     ax1.get_xaxis().set_major_locator(MaxNLocator(integer=True))
#     ax1.locator_params(nbins=5,axis='y')
#     ax1.get_xaxis().tick_bottom()
#     ax1.get_yaxis().tick_left()
#     ax1.set_ylabel(ylabel,fontsize=24)
#     ax1.set_title(r'Mean',fontsize=18)
#     plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#     ax1.set_xlim(0.9,3.1)
#     for tick in ax1.yaxis.get_major_ticks():
#         tick.label.set_fontsize(18)
#     for tick in ax1.xaxis.get_major_ticks():
#         tick.label.set_fontsize(18)
#       
#     ax2 = fig.add_subplot(122)
#     ax2.plot(data[:,0],data[:,2],'-o',c='b',lw=3.0,ms=8.0)
#     ax2.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
#     ax2.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
#     ax2.get_xaxis().set_major_locator(MaxNLocator(integer=True))
#     ax2.locator_params(nbins=5,axis='y')
#     ax2.get_xaxis().tick_bottom()
#     ax2.get_yaxis().tick_left()
#     ax2.set_title(r'Variance',fontsize=18)
#     ax2.set_xlim(0.9,3.1)
#     plt.setp([a.get_xticklabels() for a in fig.axes[:-2]], visible=False)
#     for tick in ax2.yaxis.get_major_ticks():
#         tick.label.set_fontsize(18)
#     for tick in ax2.xaxis.get_major_ticks():
#         tick.label.set_fontsize(18)
#     
#     fig.tight_layout()
# 
#     fig.savefig('BackUp_Data/SparseGrid/Tconduct.eps', format='eps')
#     plt.show()
# 
# 
# file_name1 = 'BackUp_Data/SparseGrid/level1_sparse_all.txt'
# file_name2 = 'BackUp_Data/SparseGrid/level2_sparse_all.txt'
# file_name3 = 'BackUp_Data/SparseGrid/level3_sparse_all.txt'
# data1 = np.loadtxt(file_name1)
# data2 = np.loadtxt(file_name2)
# data3 = np.loadtxt(file_name3)
# 
# n = 7
# data = np.array([ [1,data1[n,0],data1[n,1]] , [2,data2[n,0],data2[n,1]] , [3,data3[n,0],data3[n,1]]])
# 
# sparsePlot(r'$T_{conductivity}\ [Wm^{-1}K^{-1}]$')
# ax3 = fig.add_subplot(423)
# ax3.plot(data[:,0],data[:,2],'-o',c='b')
# ax3.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
# ax3.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
# ax3.get_xaxis().set_major_locator(MaxNLocator(integer=True))
# ax3.locator_params(nbins=5,axis='y')
# ax3.get_xaxis().tick_bottom()
# ax3.get_yaxis().tick_left()
# ax3.set_ylabel(r'$C_{12}\ [GPa]$')
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# ax3.set_xlim(0.9,3.1)
#   
# ax4 = fig.add_subplot(424)
# ax4.plot(data[:,0],data[:,6],'-o',c='b')
# ax4.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
# ax4.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
# ax4.get_xaxis().set_major_locator(MaxNLocator(integer=True))
# ax4.locator_params(nbins=5,axis='y')
# ax4.get_xaxis().tick_bottom()
# ax4.get_yaxis().tick_left()
# #ax4.set_ylabel(r'$C_{12}\ [GPa]$')
# ax4.set_xlim(0.9,3.1)
#    
# ax5 = fig.add_subplot(425)
# ax5.plot(data[:,0],data[:,3],'-o',c='b')
# ax5.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
# ax5.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
# ax5.get_xaxis().set_major_locator(MaxNLocator(integer=True))
# ax5.locator_params(nbins=5,axis='y')
# ax5.get_xaxis().tick_bottom()
# ax5.get_yaxis().tick_left()
# ax5.set_ylabel(r'$E_{cohesive}\ [eV]$')
# ax5.set_xlim(0.9,3.1)
#   
# ax6 = fig.add_subplot(426)
# ax6.plot(data[:,0],data[:,7],'-o',c='b')
# ax6.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
# ax6.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
# ax6.get_xaxis().set_major_locator(MaxNLocator(integer=True))
# ax6.locator_params(nbins=5,axis='y')
# ax6.get_xaxis().tick_bottom()
# ax6.get_yaxis().tick_left()
# #ax6.set_title(r'$E_{cohesive}\ [eV]$')
# ax6.set_xlim(0.9,3.1)
#   
# ax7 = fig.add_subplot(427)
# ax7.plot(data[:,0],data[:,3],'-o',c='b')
# ax7.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
# ax7.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
# ax7.get_xaxis().set_major_locator(MaxNLocator(integer=True))
# ax7.locator_params(nbins=5,axis='y')
# ax7.get_xaxis().tick_bottom()
# ax7.get_yaxis().tick_left()
# ax7.set_ylabel(r'$E_{vacancy}\ [eV]$')
# ax7.set_xlabel(r'Level')
# ax7.set_xlim(0.9,3.1)
#   
# ax8 = fig.add_subplot(428)
# ax8.plot(data[:,0],data[:,8],'-o',c='b')
# ax8.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
# ax8.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
# ax8.get_xaxis().set_major_locator(MaxNLocator(integer=True))
# ax8.locator_params(nbins=5,axis='y')
# ax8.get_xaxis().tick_bottom()
# ax8.get_yaxis().tick_left()
# ax8.set_xlabel(r'Level')
# ax8.set_xlim(0.9,3.1)
   




""" ---------- STRESS-STRAIN PLOT ----------  """
# file_name1 = 'Plots/StressStrain/1.txt'
# file_name2 = 'Plots/StressStrain/2.txt'
# file_name3 = 'Plots/StressStrain/3.txt'
# file_name4 = 'Plots/StressStrain/4.txt'
# file_name5 = 'Plots/StressStrain/5.txt'
#       
# data1 = np.loadtxt(file_name1)
# data2 = np.loadtxt(file_name2)
# data3 = np.loadtxt(file_name3)
# data4 = np.loadtxt(file_name4)
# data5 = np.loadtxt(file_name5)
#   
# fig = plt.figure(dpi=600)
# ax = fig.add_subplot(111)
# ax.plot(data1[:,10],data1[:,11]*100/3350,c='black',linewidth=3.0)
# ax.plot(data2[:,10],data2[:,11]*100/3350,c='blue',linestyle='--',linewidth=3.0)
# ax.plot(data3[:,10],data3[:,11]*100/3350,c='green',linestyle='--',linewidth=3.0)
# ax.plot(data4[:,10],data4[:,11]*50/3350,c='red',linestyle='--',linewidth=3.0)
# ax.plot(data5[:,10],data5[:,11]*50/3350,c='goldenrod',linestyle='--',linewidth=3.0)
#   
# ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
# ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
# ax.set_xlabel(r'$\epsilon$ [mm/mm]',fontsize=24)
# ax.set_ylabel(r'$\sigma$ [GPa]',fontsize=24)
# ax.set_xlim(0,0.35)
# ax.set_ylim(0,180)
# for tick in ax.yaxis.get_major_ticks():
#     tick.label.set_fontsize(18)
# for tick in ax.xaxis.get_major_ticks():
#     tick.label.set_fontsize(18)
#   
# fig.tight_layout()
# fig.savefig('Plots/StressStrain/s_s_new.eps', format='eps')
# plt.show()




