""" Leave-one-out cross-validation """

import numpy as np
import matplotlib.pyplot as plt
import subprocess


''' --- FUNCTIONS --- '''
# Trims data - Eg. trimData(data4[:,1],maxVal,minVal,True)
def trimData(data,maxVal,minVal,pltData):
    #data = data.reshape(len(data),1)
    data_trim = data[np.logical_and(data[:,10] < maxVal, data[:,10] > minVal)]
    if pltData:
        plt.figure()
        plt.scatter(range(len(data_trim)),data_trim[:,10])
        plt.grid()
    return data_trim

# Removes NaN values from any vector
def delData(data):
    #data = data.reshape(len(data),1)
    data_del = data[~np.isnan(data[:,10])]
    return data_del

''' --------------------------------------------------------------------------------------- '''

pct = '1pct'
observable = 'Tcond'
fileIn = 'InputOutput_Tcond_1pct.txt'
maxVal = 500.0
minVal = 350.0

# Creates folder name and training data set from file
filename = ''.join(['Graphene/',fileIn])
folder = ''.join(['Graphene/',pct,'/Loo/'])
training = np.genfromtxt(filename,delimiter='\t',filling_values=np.NaN)

# Deleting NaN values and trimming data
training = delData(training)
training = trimData(training,maxVal,minVal,False)

'''Testing with only 10 realizations...'''
# training = np.delete(training,np.s_[99::1], 0)

# Number of realizations and inputs (last input is an observable value)
npts,nvars = np.shape(training)
ninputs = nvars - 1

# Getting min and max value for each variable
lb = np.min(training,0)
ub = np.max(training,0)

# Creating an information file to be pass to the filter driver (Gaussian process model)
fp = open('gppar.dat','w')
fp.write('%d\n' %ninputs)
fp.write('%d\n' %(npts-1))
fp.write('1\n')
fp.write('0\n')
fp.write('2\n')
fp.write('trainfile\n')
fp.write('testfile\n')
fp.close()

# Scaling the inputs from the training data
Sample = np.zeros((npts,nvars))
for i in range(ninputs):
    Sample[:,i] = (training[:,i]-lb[i])/(ub[i]-lb[i])
Sample[:,-1] = training[:,-1]

np.savetxt(''.join([folder,'sample_',observable,'.dat']),Sample)
assert Sample.shape == training.shape, 'Error in dimensions of the data set'

# Leave-one-out process
training = np.zeros((npts-1,nvars))     # Training set contains npts-1 realizations (leaving one out)
testing = np.zeros((1,ninputs))
actual = np.zeros((npts,1))
predicted = np.zeros((npts,1))
vpredicted = np.zeros((npts,1))
for i in range(npts):
    if i == 0:
        training = Sample[i+1:npts,:]
        testing = Sample[i,0:ninputs]           # Testing has the inputs from the left out set
        actual[i] = Sample[i,ninputs]           # Saving actual observable
        np.savetxt('trainfile',training)
        np.savetxt('testfile',testing.reshape(1,ninputs))
        subprocess.call('./filter')             # Calling the filter driver & loading results
        filter_out = np.loadtxt('filterout.dat')
        predicted[i] = filter_out[0]
        vpredicted[i] = filter_out[1]
    elif i == npts-1:
        training = Sample[0:npts-1,:]
        testing = Sample[i,0:ninputs]           # Testing has the inputs from the left out set
        actual[i] = Sample[i,ninputs]           # Saving actual observable
        np.savetxt('trainfile',training)
        np.savetxt('testfile',testing.reshape(1,ninputs))
        subprocess.call('./filter')             # Calling the filter driver & loading results
        filter_out = np.loadtxt('filterout.dat')
        predicted[i] = filter_out[0]
        vpredicted[i] = filter_out[1]
    else:
        training[0:i,:] = Sample[0:i,:]
        training[i:npts-1,:] = Sample[i+1:npts,:]
        testing = Sample[i,0:ninputs]           # Testing has the inputs from the left out set
        actual[i] = Sample[i,ninputs]           # Saving actual observable
        np.savetxt('trainfile',training)
        np.savetxt('testfile',testing.reshape(1,ninputs))
        subprocess.call('./filter')             # Calling the filter driver & loading results
        filter_out = np.loadtxt('filterout.dat')
        predicted[i] = filter_out[0]
        vpredicted[i] = filter_out[1]
    training = np.zeros((npts-1,nvars))

# Calculating the standardized cross validation residuals (SCVR)
scvr = np.zeros((npts))
for i in range(npts):
    scvr[i] = (actual[i]-predicted[i])/np.sqrt(vpredicted[i])

# Plotting the leave-one-out
fig = plt.figure(1)
ax1 = fig.add_subplot(111)
ax1.scatter(actual.reshape(npts,1),predicted.reshape(npts,1))
ax1.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
ax1.get_xaxis().tick_bottom()
ax1.get_yaxis().tick_left()
ax1.set_title(r'Leave-one-out')
ax1.set_xlabel(r'Actual')
ax1.set_ylabel(r'Predicted')
fig.tight_layout()
fig.savefig(''.join([folder,'loo_',observable,'.eps']), format='eps')

# Plotting SCVR
a = np.arange(0,len(scvr),1)
   
fig = plt.figure(2)
ax1 = fig.add_subplot(111)
ax1.scatter(a,scvr)
ax1.axhline(y=3,xmin=0,xmax=1,c="r",linewidth=1,zorder=0,linestyle='--')
ax1.axhline(y=-3,xmin=0,xmax=1,c="r",linewidth=1,zorder=0,linestyle='--')
ax1.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
ax1.get_xaxis().tick_bottom()
ax1.get_yaxis().tick_left()
ax1.set_ylabel(r'SCVR')
ax1.set_xlabel(r'Sample Points')
ax1.set_xlim(0,len(a))
ax1.set_xticklabels('')
ax1.set_ylim(-5,5)
fig.tight_layout()
fig.savefig(''.join([folder,'scvr_',observable,'.eps']), format='eps')

# Saving files
model_loo = np.hstack((actual.reshape(npts,1),predicted.reshape(npts,1)))
np.savetxt(''.join([folder,'loo_',observable,'.dat']),model_loo)
np.savetxt(''.join([folder,'scvr_',observable,'.dat']),scvr.reshape(npts,1))

plt.show()