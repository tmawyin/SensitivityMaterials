from ase.calculators.eam import EAM
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as ip
import random as rdm
import design
import pypeaks as pp

'''---------------PLOTTING---------------'''
# PlotFunctions: plots the original potential functions (compares it with ASE.EAM plot function)
def PlotFunctions():
    # Plotting original data F, rho, phi respectively 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(rhos,F_rho,s=2.0,marker='o',color='b')
    ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
    ax.set_xlabel(r'$\rho$',fontsize=20)
    ax.set_ylabel(r'F($\Sigma\rho$)',fontsize=20)
    #ax.set_title(r'Embedding Function',fontsize=18)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(18)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(18)
    fig.tight_layout()
    fig.savefig('EmbeddingFunction.eps', format='eps')
      
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(rs,rho_r,s=2.0,marker='o',color='b')
    ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
    ax.set_xlabel(r'$r$',fontsize=20)
    ax.set_ylabel(r'$\rho$(r)',fontsize=20)
    #ax.set_title(r'Electron Density Function',fontsize=18)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(18)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(18)
    fig.tight_layout()
    fig.savefig('ElectronDensity.eps', format='eps')
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(rs[1:],r_phi[0,0][1:]/rs[1:],s=2.0,marker='o',color='b')
    ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
    ax.set_xlabel(r'$r$',fontsize=20)
    ax.set_ylabel(r'$\phi$(r)',fontsize=20)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(18)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(18)
    # ax.set_title(r'Pair Potential Interaction')
    fig.tight_layout()
    fig.savefig('PairPotential.eps', format='eps')
    
    # Plotting interpolation from ASE-EAM module
    fig = plt.figure()
    mishin.plot()
    plt.ylim(-10,70)
    fig.tight_layout()
    #fig.savefig('Interpolation.eps', format='eps')

'''---------------SAVING FILE---------------'''
# save2File: Saves all the data from the original potential into columns
def save2File():
    ''' 
    Saving functions to file in the following format:
    Column 1 - r
    Column 2 - rho
    Column 3 - F(rho)
    Column 4 - r*Phi
    Column 5 - Rho(r)
    '''
    rs = rs.reshape((len(rs),1))
    rhos = rhos.reshape((len(rhos),1))
    r_phi = r_phi[0,0].reshape((len(r_phi[0,0]),1))
      
    data = np.hstack((rs,rhos,F_rho.T,r_phi,rho_r.T))
    np.savetxt('Al_EAM_datapoints.dat',data,fmt='%1.16e') 

'''---------------Retrieve Peaks---------------'''
def retrievePeaks(x,y,n,m):
    # Breaking down the function using n spaced points 
    x_temp = x[0::n]
    y_temp = y[0::n]
    # Adding end point values
    x_temp = np.append(x_temp, x[-1])
    y_temp = np.append(y_temp, y[-1])
    # Interpolation to get error
    f_intp = ip.UnivariateSpline(x_temp,y_temp,s=0)
    # Absolute error
    f_eval = f_intp(x)
    error = np.abs((y-f_eval))
    # Error data peaks
    data = pp.Data(x,error)
    data.get_peaks()
    pks = np.asarray(data.peaks.get('peaks')).T
    # Sorting from lowest peak to highest
    pks = pks[pks[:,1].argsort()]
    # Getting only m peaks (last m since list is in ascending order)
    pks = pks[-m::]
    # Determine the coefficients
    pkCoef = np.asarray(sorted([np.where(x==i)[0][0] for i in pks[:,0]]))
    # Insert new coefficients in their respective places
    loctn = np.ceil(pkCoef/float(n))
    x_temp = np.insert(x_temp, loctn, x[pkCoef])
    y_temp = np.insert(y_temp, loctn, y[pkCoef])
    return x_temp,y_temp

'''---------------WRITE EAM---------------'''
# Writes the EAM potential to a file given the interpolated function and the file name
def writeEAM(fint,fname,numFun):
    f = open(fname,'w')
    # Header lines
    f.write(' Al EAM from Phys. Rev. B 59, 3393 (1999) in the LAMMPS setfl format.\n')
    f.write(' Conversion by C. A. Becker from Y. Mishin files.\n')
    f.write(' Modified to introduce uncertainty by Tomas Mawyin.\n')
    # Elements
    f.write('%d\t %s \n' %(Nelement,Element1))   # Nelement, Element1...
    # Dimensions of the system
    f.write('%d\t %0.15E\t %d\t %0.15E\t %0.15E \n' %(Nrho,drho,Nr,dr,cutoff))
    # Properties of elements
    f.write('%d\t %0.15E\t %0.15E\t %s \n' %(Z,mass,lattice,lat_type))
    # Arrays of data
    F = fint[0](rhos) if numFun == 1 else F_rho[0] 
    Rho = fint[1](rs) if numFun == 2 else rho_r[0]
    rPhi = rs*fint[2](rs) if numFun == 3 else r_phi[0,0][:]
    if numFun == 4:
        F = fint[0](rhos)
        Rho = fint[1](rs)
        rPhi = rs*fint[2](rs)
    if numFun == 5:
        F = fint[0](rhos)
        Rho = fint[1](rs)
    if numFun == 6:
        F = fint[0](rhos)
        rPhi = rs*fint[2](rs)
    np.savetxt(f, F, fmt='%+0.18E')
    np.savetxt(f, Rho, fmt='%+0.18E')
    np.savetxt(f, rPhi, fmt='%+0.18E')
    f.close

'''---------------INTERPOLATION---------------'''
# FuncInterpol: will plot the interpolation and error of y(x) based on n-spaced points. Returns the interpolation handle
def FuncInterpol(x,y,n,plot,title,saving):
    # Breaking down the function using n spaced points 
    x_temp = x[0::n]
    y_temp = y[0::n]
    # Adding end point values    
#     x_temp = np.append(x_temp, x[-1])
#     y_temp = np.append(y_temp, y[-1])
    
    # Alternative to get points by adding more points based on error peaks (m peaks)
    # m = 2
    # x_temp, y_temp = retrievePeaks(x,y,n,m)
    
    # Using univariate spline with s=0 (similar to interpolated univariate spline) -> Good fit
    f_intp = ip.UnivariateSpline(x_temp,y_temp,s=0)
    
    # Saving coefficients of interpolation
    f_coef = f_intp.get_coeffs()    
    np.savetxt('Coefficients/'+title+'_originalCoeff.txt',f_coef,fmt='%1.16e',newline="\t")
    
    if plot:
        f_eval = f_intp(x)
        # Plotting interpolation against original curve and error
        fig = plt.figure()
        ax = fig.add_subplot(211)
        # Original curve
        ax.plot(x,y,'k',label='Original Curve')
        # Interpolated curve 
        ax.plot(x,f_eval,'--b',label='Interpolated Curve')
        ax.plot(x[0::n],f_eval[0::n],'s')
        ax.set_ylabel(r'Function Evaluation')
        ax.grid()
        ax.legend()
        ax.set_title(title)
        # Error plot
        ax_e = fig.add_subplot(212)
        ax_e.plot(x,np.abs((y-f_eval)),'r')
#         ax_e.set_yscale('log')
        ax_e.set_ylabel(r'Interpolation error')
        ax_e.grid()
        fig.tight_layout()
    if saving: # Used for Greedy Algorithm
        # Saving training data 
        x_svg = np.reshape(x_temp,(len(x_temp),1))
        y_svg = np.reshape(y_temp,(len(y_temp),1))
        data = np.hstack((x_svg,y_svg))
        np.savetxt(title+'_train.txt', data,fmt='%1.16e')
        # Saving testing and prediction data
        x_tst = np.reshape(x,(len(x),1))
        y_tst = np.reshape(y,(len(y),1))
        np.savetxt(title+'_test.txt', x_tst,fmt='%1.16e')
        np.savetxt(title+'_predict.txt', y_tst,fmt='%1.16e')
        
    # Return interpolation handle
    return f_intp

'''---------------UNCERTAINTY---------------'''
# FuncUncertainty: gets an interpolation handle and a variability(in %) to set coefficients uncertainty. Returns new handle and new coefficients
def FuncUncertainty(fint,var,opt):
    v = var/100.0
    f_coef = fint.get_coeffs()
    minCoef = f_coef*(1.0-v)
    maxCoef = f_coef*(1.0+v)
    # Uniform distribution using variability
    new_coef = rdm.uniform(minCoef,maxCoef)
    # Saving new coefficients
    k,m = fint._data[5],fint._data[7]
    fint._data[9][:m-k-1] = new_coef
    if len(opt)>1:
        fig = plt.figure()
        ax = fig.add_subplot(211)
        # Original curve for phi(r)
        ax.plot(opt[0],opt[1],'k',label='Original Curve')
        ax.plot(opt[0],fint(opt[0]),'--b',label='Uncertainty Curve')
        ax.set_ylabel(r'Function Evaluation')
        ax.grid()
        ax.legend()
        # Error plot
        ax_e = fig.add_subplot(212)
        ax_e.plot(opt[0],np.abs((opt[1]-fint(opt[0]))),'r')
        ##ax_e.set_yscale('log')
        ax_e.set_ylabel(r'Interpolation error')
        ax_e.grid()
        fig.tight_layout()
    return fint, new_coef

'''---------------MC Technique---------------'''
# MCpotential: uses MC technique to create m many potentials with variability var based on the interpolation fint
def MCpotential(fint,var,m,numFun):
    # for i in range(m):
    F_int,F_coef = FuncUncertainty(fint[0],var,[])
    Rho_int,Rho_coef = FuncUncertainty(fint[1],var,[])
    rPhi_int,rPhi_coef = FuncUncertainty(fint[2],var,[])
    writeEAM([F_int,Rho_int,rPhi_int],'MCPot/Al99.eam.%d.alloy'%(m),numFun)
    
    data = F_coef if numFun ==1 else Rho_coef if numFun ==2 else rPhi_coef if numFun == 3 else np.hstack((F_coef,Rho_coef,rPhi_coef)) if numFun == 4 else np.hstack((F_coef,Rho_coef)) if numFun == 5 else np.hstack((F_coef,rPhi_coef)) 
    np.savetxt('Coefficients/InputCoeff_%d.txt'%(i),data,fmt='%1.16e',newline="\t")
        
         

'''---------------SparseGrid Technique---------------'''
def SGpotential(fint,level,var):
    v = var/100.0
    # Coefficients for F(rho)
    f0 = fint[0].get_coeffs()
    k0,m0 = fint[0]._data[5],fint[0]._data[7]
    minCoef0 = f0*(1.0-v)
    maxCoef0 = f0*(1.0+v)
    
    # Coefficients for Rho(r)
    f1 = fint[1].get_coeffs()
    k1,m1 = fint[1]._data[5],fint[1]._data[7]
    minCoef1 = f1*(1.0-v)
    maxCoef1 = f1*(1.0+v)
    
    # Coefficients for r*Phi(r)
    f2 = fint[2].get_coeffs()
    k2,m2 = fint[2]._data[5],fint[2]._data[7]
    minCoef2 = f2*(1.0-v)
    maxCoef2 = f2*(1.0+v)
    
    # Setting up sparse grid
    ndim = len(f0)              # Changes based on the number of coefficients from interpolation
    X,w = design.sparse_grid(ndim, level, rule='CC')
    X0 = np.copy(X); X1 = np.copy(X); X2 = np.copy(X)   # Copying sparse grid to scale it according to interpolation
    npoints = len(X)
    # Scaling nodes (each column in X represent a dimension with npoints values)
    for i in range(ndim):
        X0[:,i] = 0.5*(maxCoef0[i]-minCoef0[i])*X[:,i] + 0.5*(maxCoef0[i]+minCoef0[i])
        X1[:,i] = 0.5*(maxCoef1[i]-minCoef1[i])*X[:,i] + 0.5*(maxCoef1[i]+minCoef1[i])
        X2[:,i] = 0.5*(maxCoef2[i]-minCoef2[i])*X[:,i] + 0.5*(maxCoef2[i]+minCoef2[i])
    # Scaling weights and saving to file
    for i in range(npoints):
        w[i] = w[i]*(0.5)**ndim
        fint[0]._data[9][:m0-k0-1] = X0[i,:]
        fint[1]._data[9][:m1-k1-1] = X1[i,:]
        fint[2]._data[9][:m2-k2-1] = X2[i,:]
        writeEAM([fint[0],fint[1],fint[2]],'SGPot/Al99.eam.%d.alloy'%(i))
    return w


            
''' ---------- ALUMINIUM EAM POTENTIAL ---------- '''
# Al PROPERTIES
mishin = EAM(potential='Al99.eam.alloy')
Nelement = len(mishin.elements)
Element1 = mishin.elements[0]
Nrho = mishin.nrho
drho = mishin.drho
Nr = mishin.nr
dr = mishin.dr
cutoff = mishin.cutoff
Z = mishin.Z[0]
mass = mishin.mass[0]
lattice = mishin.a[0]
lat_type = mishin.lattice[0]

# Original data from Al potential file
F_rho = mishin.embedded_data    # F(rho) data
r_phi = mishin.rphi_data        # r*phi(r) data
rho_r = mishin.density_data     # rho(r) data

# r values -> ranging from 0 to cutoff-dr
rs = np.arange(0, Nr)*(dr)
# rho values -> as the sum of rho's for embedding function
rhos = np.arange(0, Nrho)*(drho)
# Getting phi from r*phi in data file
phi = r_phi[0,0][1:]/rs[1:]
phi = np.insert(phi,0,0)

# PlotFunctions()

''' *********** FUNCTION CALLS *********** '''

n0 = 1000        # Interpolate using n-spaced points
n1 = 800
n2 = 400
var = 2.5       # Add uncertainty with variability (var in [%])
numFun = 1      # Choose: 1 for F(rho), 2 for Rho(r), 3 for Phi(r), 4 for ALL, 5 for F(rho) and Rho(r), 6 gor F(rho) and Phi(r)  

# ----- F(rho) INTERPOLATION
# rho_intp = rhos         # Reshaping array
# F_intp = F_rho[0]
# f0 = FuncInterpol(rho_intp,F_intp,n0,True,r'F($\rho$)',True)           # Interpolation without uncertainty
# f0_uncert = FuncUncertainty(f0,var,[rho_intp,F_intp])       # Adding uncertainty to interpolation

# ----- RHO(r) INTERPOLATION
# r_rho_intp = rs         # Reshaping array
# rho_intp = rho_r[0]
# f1 = FuncInterpol(r_rho_intp,rho_intp,n1,False,r'$\rho$(r)',False)        # Interpolation without uncertainty
# f1_uncert = FuncUncertainty(f1,var,[r_rho_intp,rho_intp])   # Adding uncertainty to interpolation

# ----- PHI(r) INTERPOLATION
# r_phi_intp = rs[1::]    # Reshaping array
# phi_intp = phi[1::]
# f2 = FuncInterpol(r_phi_intp,phi_intp,n2,False,r'$\phi$(r)',False)        # Interpolation without uncertainty
# f2_uncert = FuncUncertain0/10.ty(f2,var,[r_phi_intp,phi_intp])   # Adding uncertainty to interpolation

# writeEAM([f0,f1,f2],'Al99.eam.intp.alloy',numFun)

# ----- MONTE CARLO
nPot = 501
for i in range(nPot):
    # Function F(rho)
    rho_intp = rhos         # Reshaping array
    F_intp = F_rho[0]
    f0 = FuncInterpol(rho_intp,F_intp,n0,False,r'F($\rho$)',False)           # Interpolation without uncertainty
    
    # Function Rho(r)
    r_rho_intp = rs         # Reshaping array
    rho_intp = rho_r[0]
    f1 = FuncInterpol(r_rho_intp,rho_intp,n1,False,r'$\rho$(r)',False)        # Interpolation without uncertainty
    
    # Function Phi(r)
    r_phi_intp = rs[1::]    # Reshaping array
    phi_intp = phi[1::]
    f2 = FuncInterpol(r_phi_intp,phi_intp,n2,False,r'$\phi$(r)',False)        # Interpolation without uncertainty

    MCpotential([f0,f1,f2],var,i,numFun)

# ----- SPARSE GRID
# SGpotential([f0,f1,f2],1,0)


# print len(f0.get_coeffs())
# print len(f1.get_coeffs())
# print len(f2.get_coeffs())
# print "total number of coefficients: ", (len(f0.get_coeffs())+len(f1.get_coeffs())+len(f2.get_coeffs()))
# plt.show()
# '''
