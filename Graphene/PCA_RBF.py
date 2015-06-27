''' SAOlib for Graphene and Metals '''

# import cPickle as pickle
import numpy as np
import saolib
import time
import matplotlib.pyplot as plt

''' ---------------------- FUNCTIONS ---------------------- '''
# Trims data - Eg. trimData(data4[:,1],maxVal,minVal,True)
def trimData(data,maxVal,minVal,pltData):
    #data = data.reshape(len(data),1)
    data_trim = data[np.logical_and(data[:,-1] < maxVal, data[:,-1] > minVal)]
    if pltData:
        plt.figure()
        plt.scatter(range(len(data_trim)),data_trim[:,-1])
        plt.grid()
    return data_trim

# Removes NaN values from any vector
def delData(data):
    #data = data.reshape(len(data),1)
    data_del = data[~np.isnan(data[:,-1])]
    return data_del

''' --------------------------------------------------------------------------------------- '''

pct = '5pct'
observable = 'Estacking'
fileIn = 'Stacking_5pct.txt'
# fileIn = 'Input_Output_1percent.txt'
maxVal = 130.0
minVal = 90.0
    
# Creates folder name and training data set from file: GRAPHENE
# filename = ''.join(['Graphene/',fileIn])
# folder = ''.join(['Graphene/',pct,'/'])
# Creates folder name and training data set from file: METALS
filename = ''.join(['Metals/FPhi/',fileIn])
folder = ''.join(['Metals/FPhi/',pct,'/'])

# training = np.loadtxt(filename)
training = np.genfromtxt(filename,delimiter='\t',filling_values=np.NaN)
# training = np.delete(training,24,1) # METAL: Uncomment for F and Rho - last variable is zero which causes problems
np.random.shuffle(training)

# Deleting NaN values and trimming data
# training = delData(training)
# training = trimData(training,maxVal,minVal,False)
    
# Saving to X and Y vectors
X = training[:,:-1].T.reshape(training.shape[1]-1,training.shape[0])
Y = training[:,-1].T.reshape(1,training.shape[0])
print Y.shape
print X.shape
# number of input variables
nvar = np.size(X,0)
# number of realizations
mpts = np.size(X,1)
    
# Select a train size - usually -> int(max(0.8*mpts,10*nvar))
nTrain = 25
# Test points - difference between number of points and training points 
nTest = mpts - nTrain
# Number of observables - only 1 observable at a time
kresp = np.size(Y,0)
 
# Retrieve options - either PCARBF or RBF
options = saolib.getOptions('PCARBF')
# options = saolib.getOptions('RBF')
     
# Train saolib model and compute training time
print 'Retrieved saolib options... training model'
    
t0 = time.time()
Model = saolib.trainModel(X[:,0:nTrain],Y[:,0:nTrain],options)
t1 = time.time()
print 'Time to train model: %f' %(t1-t0)
     
# Store saolib model
# print Model['pcaModel']
# print Model['pcaModel']['basesInp']
# saolib.saveModel(Model,'MolecularDynamics.model')

# Evaluate testing set
print '... evaluating model'
t0 = time.time()
Yhat = saolib.evalModel(Model,X[:,nTrain:mpts])
t1 = time.time()
print 'Time to eval model: %f' %(t1-t0)

# Store error norm of testing set
print 'Storing testing set error norms to file'
errNorms = ((Yhat-Y[0,nTrain:mpts])/Y[0,nTrain:mpts])*100.0
# np.savetxt('MolecularDynamics_ERR.txt', errNorms, delimiter=",", fmt="%1.8e")

from matplotlib.ticker import MaxNLocator
fig = plt.figure(dpi=600)
ax = fig.add_subplot(121)
ax.plot(Y[0,nTrain:mpts],np.transpose(Yhat),'o',ms=8.0)
ax.plot([np.min(Yhat),np.max(Yhat)],[np.min(Yhat),np.max(Yhat)], ls="--", c="red",lw=2.0)
ax.set_xlabel(r'Actual Values',fontsize=16)
ax.set_ylabel(r'Predicted Values',fontsize=16)
ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
# ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
ax.locator_params(nbins=5,axis='x')
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(14)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(14)
    
ax2 = fig.add_subplot(122)
ax2.plot(np.arange(0,nTest),np.transpose(errNorms),'o',ms=8.0)
ax2.set_xlabel(r'Testing Points',fontsize=16)
ax2.set_ylabel(r'Error',fontsize=16)
ax2.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
ax2.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.8)
ax2.get_xaxis().tick_bottom()
ax2.get_yaxis().tick_left()
ax2.get_xaxis().set_major_locator(MaxNLocator(integer=True))
ax2.locator_params(nbins=3,axis='x')
ax2.set_xlim(0.0,400)
for tick in ax2.yaxis.get_major_ticks():
    tick.label.set_fontsize(14)
for tick in ax2.xaxis.get_major_ticks():
    tick.label.set_fontsize(14)

fig.tight_layout()
fig.savefig(''.join([folder,'PCARBF_',observable,'.eps']),format='eps')

# plt.show()

