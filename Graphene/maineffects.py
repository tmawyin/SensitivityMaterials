""" Computing main effects & sensitivity """

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import subprocess

''' Data Trimming '''
# Trims data - Eg. trimData(data4[:,1],maxVal,minVal,True)
def trimData(data,maxVal,minVal,pltData):
    data_trim = data[np.logical_and(data[:,10] < maxVal, data[:,10] > minVal)]
    if pltData:
        plt.figure()
        plt.scatter(range(len(data_trim)),data_trim[:,10])
        plt.grid()
    return data_trim

''' Remove NaN values '''
def delData(data):
    data_del = data[~np.isnan(data[:,10])]
    return data_del

''' Latin Hypercube Sampling '''
def lhsamp(m,n,lb,ub):
    S = np.zeros((m,n))
    for i in range(n):
        S[:,i] = (npr.rand(m,1) + npr.permutation(m).reshape(m,1)).reshape(m,) / m

    for i in range(n):
        S[:,i] = lb[i] + S[:,i]*(ub[i]-lb[i]) 
    return S

''' --------------------------------------------------------------- '''

pct = '1pct'
observable = 'E59_prev'
fileIn = 'InputOutput_E59_1pct_prev.txt'
maxVal = 10.0
minVal = 5.5

# Creates folder name and training data set from file
filename = ''.join(['Graphene/',fileIn])
folder = ''.join(['Graphene/',pct,'/MainEffects/'])
training = np.genfromtxt(filename,delimiter='\t',filling_values=np.NaN)

# Deleting NaN values and trimming data
# training = delData(training)
training = trimData(training,maxVal,minVal,False)

# Number of realizations and inputs (last input is an observable value)
npts,nvars = np.shape(training)
ninputs = nvars - 1    # Number of actual inputs 

nsamp = 10000    # Number of LHS samples used to approximate integrals
ncalc = 5    # Number of points at which the main effects are calculated

# Getting min and max value for each variable
lb = np.min(training,0)
ub = np.max(training,0)

# Scaling the inputs from the training data from [0-1]
Sample = np.zeros((npts,nvars))
trainfile = np.zeros((npts,nvars))
for i in range(ninputs):
    Sample[:,i] = (training[:,i]-lb[i])/(ub[i]-lb[i])
Sample[:,-1] = training[:,-1]
np.savetxt(''.join([folder,'sample_',observable,'.dat']),Sample)
assert Sample.shape == training.shape, 'Error in dimensions of the data set'

trainfile = Sample
np.savetxt('trainfile',trainfile)
np.savetxt('testfile',Sample)

# Creating an information file to be pass to the filter driver (Gaussian process model)
fp = open('gppar.dat','w')
fp.write('%d\n' %ninputs)   # Number of inputs
fp.write('%d\n' %npts)      # Number of realizations
fp.write('%d\n' %npts)      # Number of testing points
fp.write('0\n')
fp.write('2\n')
fp.write('trainfile\n')
fp.write('testfile\n')
fp.close()

# Initial model training
subprocess.call('./filter')

# Set optimization-flag to zero for subsequent runs
fp = open('gppar.dat','w')
fp.write('%d\n' %ninputs)
fp.write('%d\n' %npts)
fp.write('%d\n' %nsamp)
fp.write('0\n')
fp.write('0\n')
fp.write('trainfile\n')
fp.write('testfile\n')
fp.close()

X = np.zeros((ncalc,ninputs))
Y_mean = np.zeros((ncalc,ninputs))
Y_var = np.zeros((ncalc,ninputs))

S = np.zeros((nsamp,ninputs))
lb = np.zeros((ninputs,1))
ub = np.ones((ninputs,1))
var_sensitivity = np.zeros((ninputs,1))
sensitivity = np.zeros((ninputs,1))

label = np.char.array([r'$Q_{CC}$',r'$\alpha_{CC}$',r'$\beta_{CC1}$',r'$\beta_{CC2}$',r'$\beta_{CC3}$',
                       r'$rc^{LJmin}_{CC}$',r'$rc^{LJmax}_{CC}$',r'$\epsilon_{CC}$',r'$\sigma_{CC}$',r'$\epsilon^{T}_{CCCC}$'])
fig = plt.figure(1)     # Generate the main effect plot
for i in range(ninputs):
    S = lhsamp(nsamp,ninputs,lb,ub)
    temp1 = np.linspace(0,1,ncalc)
    X[:,i] =  temp1.reshape(ncalc,)
    for j in range(ncalc):
        temp2 = temp1[j]*np.ones((nsamp,1))
        testing = S
        testing[:,i] = temp2.reshape(nsamp,)
        np.savetxt('testfile',testing)
        subprocess.call('./filter')
        filter_out = np.loadtxt('filterout.dat')
        Y_mean[j,i] = np.mean(filter_out[:,0])
        Y_var[j,i] = np.mean(filter_out[:,1])

    var_sensitivity[i,0] = np.var(Y_mean[:,i])
    sensitivity = var_sensitivity/np.sum(var_sensitivity)
    
    ax = fig.add_subplot(ninputs/2,2,i+1)
    ax.plot(X[:,i],Y_mean[:,i],'b',linewidth=2)
    ax.plot(X[:,i],Y_mean[:,i]+2*np.sqrt(Y_var[:,i]),'--r')
    ax.plot(X[:,i],Y_mean[:,i]-2*np.sqrt(Y_var[:,i]),'--r')
    ax.set_xlabel(label[i])
    plt.locator_params(nbins=5)
    
# fig.suptitle(r'$1\%$ Variability',fontsize=11)
fig.tight_layout()
fig.subplots_adjust(top=0.92,hspace=0.5)
plt.setp([a.get_xticklabels() for a in fig.axes[:-2]], visible=False)
fig.savefig(''.join([folder,'MainEffect_',observable,'.eps']), format='eps')

# Saving files
mainEff = np.hstack((X[:,0].reshape(ncalc,1), Y_mean, 2*np.sqrt(Y_var)))
np.savetxt(''.join([folder,'mEffect_',observable,'.dat']),mainEff)
np.savetxt(''.join([folder,'Sensitivity_',observable,'.dat']),sensitivity)

# Plotting pie chart
colors = ['lightcoral','red','gold','lightskyblue','white','yellowgreen','blue',
          'pink', 'darkgreen','yellow','grey','violet','magenta','cyan'] 
p = np.loadtxt(''.join([folder,'Sensitivity_',observable,'.dat']))
pct = 100.*p
explode = [0.05]*len(p)

fig_pie = plt.figure(2)
ax_pie = fig_pie.add_axes([0.25, 0.01, 0.8, 0.95])
patches, texts = ax_pie.pie(p, colors=colors, startangle=90, radius=1, explode=explode)
labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(label, pct)]
legd = ax_pie.legend(patches, labels, loc='center', bbox_to_anchor=(-0.13, 0.5), fontsize=11,borderaxespad=0.)
fig_pie.savefig(''.join([folder,'PieChart_',observable,'.eps']), format='eps', bbox_extra_artist=(legd), bbox_inches='tight')
plt.show()
