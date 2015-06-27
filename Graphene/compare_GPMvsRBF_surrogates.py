""" Compare Gaussian Process and Radial Basic Functions surrogate models using K-fold """

import numpy as np
import matplotlib.pyplot as plt
import subprocess
import saolib
import time


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

pct = '5pct'
observable = 'C11'
fileIn = 'InputOutput_C11_05pct.txt'
maxVal = 1500
minVal = 1000.0
# Number of training points
nTrain = 250

# Creates folder name and training data set from file
filename = ''.join(['Graphene/',fileIn])
folder = ''.join(['Graphene/',pct,'/ModelCompare/'])
training = np.genfromtxt(filename,delimiter='\t',filling_values=np.NaN)
# training = np.loadtxt(filename)

# Deleting NaN values and trimming data
training = delData(training)
training = trimData(training,maxVal,minVal,False)
# Saving the full data just in case!
Fulldata = training

# Number of realizations and inputs (last input is an observable value)
npts,nvars = np.shape(training)
ninputs = nvars - 1

# Break down training and testing test - randomly
nTest = npts - nTrain

# Shuffling the array to make it random
np.random.shuffle(training)

''' --- RBF model --- '''
# Save X (inputs) and Y (outputs) vectors for SAOlib functionality
X = training[:,:-1].T.reshape(10,training.shape[0])
Y = training[:,10].T.reshape(1,training.shape[0])

# Retrieve options - RBF
t0 = time.time()
options = saolib.getOptions('RBF')
Model = saolib.trainModel(X[:,0:nTrain],Y[:,0:nTrain],options)
Yhat_rbf = saolib.evalModel(Model,X[:,nTrain:npts])
t1 = time.time()
print 'RBF MODEL TIME: %f' %(t1-t0)

''' --- GPM model --- '''
# Getting min and max value for each variable
lb = np.min(training,0)
ub = np.max(training,0)

# Creating an information file to be pass to the filter driver (Gaussian process model)
fp = open('gppar.dat','w')
fp.write('%d\n' %(ninputs)) # Number of inputs
fp.write('%d\n' %(nTrain))  # Number of training points 
fp.write('%d\n' %(nTest))   # Number of testing points
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
np.savetxt('trainfile',Sample[0:nTrain,:])
np.savetxt('testfile',Sample[nTrain:npts,:-1])

# Calling GPM model and storing results
Yhat_gpm = np.zeros((nTest,1))
t0 = time.time()
subprocess.call('./filter')             # Calling the filter driver & loading results
t1 = time.time()
filter_out = np.loadtxt('filterout.dat')
Yhat_gpm = filter_out[:,0]
print 'GPM MODEL TIME: %f' %(t1-t0)

''' --- Plotting results --- '''
fig = plt.figure(dpi=600)
ax = fig.add_subplot(111)
ax.plot(Y[0,nTrain:npts],np.transpose(Yhat_rbf),'s',ms=5.0,c='green',label='RBF')
ax.plot(Y[0,nTrain:npts],np.transpose(Yhat_gpm),'o',ms=5.0,c='blue',label='GPM')
ax.plot([minVal,maxVal],[minVal,maxVal], '--m', linewidth=2.0)
ax.set_xlabel(r'Actual Values',fontsize=16)
ax.set_ylabel(r'Predicted Values',fontsize=16)
ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.legend(loc=2,numpoints=1)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(14)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(14)
fig.tight_layout()
fig.savefig(''.join([folder,'kfold_',observable,'.eps']), format='eps')
plt.show()

# from matplotlib.ticker import MaxNLocator
# ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
# ax.locator_params(nbins=3,axis='x')
# ax1.plot(actual,predicted,'o',ms=6.0,label='GPM')
# ax1.plot(actual,Yhat1,'s',ms=6.0,c='green',label='RBF')
# ax.plot([np.min(Yhat),np.max(Yhat)],[np.min(Yhat),np.max(Yhat)], ls="--", c="red",lw=2.0)
