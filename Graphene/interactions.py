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
observable = 'Ecoh'
fileIn = 'InputOutput_Ecoh_1pct.txt'
maxVal = -5.0
minVal = -10.0

# Creates folder name and training data set from file
# filename = ''.join(['../Graphene/',fileIn])
# folder = ''.join(['../Graphene/',pct,'/Interactions/'])
#training = np.genfromtxt(filename,delimiter='\t',filling_values=np.NaN)
training = np.loadtxt('training_old.dat')
print training.shape
# Deleting NaN values and trimming data
# training = delData(training)
# training = trimData(training,maxVal,minVal,False)

# Number of realizations and inputs (last input is an observable value)
npts,nvars = np.shape(training)
ninputs = nvars - 1    # Number of actual inputs 

nsamp = 5000   # Number of LHS samples used to approximate integrals
ncalc = 10      # Number of points at which the main effects are calculated

# Getting min and max value for each variable
lb = np.min(training,0)
ub = np.max(training,0)

# Scaling the inputs from the training data from [0-1]
Sample = np.zeros((npts,nvars))
trainfile = np.zeros((npts,nvars))
for i in range(ninputs):
    Sample[:,i] = (training[:,i]-lb[i])/(ub[i]-lb[i])
# Adding the output value
Sample[:,-1] = training[:,-1]
# np.savetxt(''.join([folder,'sample_',observable,'.dat']),Sample)
assert Sample.shape == training.shape, 'Error in dimensions of the data set'

# Saving normalized data + output to the training and testing file
trainfile = Sample
np.savetxt('trainfile',trainfile)
np.savetxt('testfile', trainfile)

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

''' Testing Model '''
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

# Matrices to store values and mesh points
x = np.linspace(0, 1, ncalc)
y = np.linspace(0, 1, ncalc)
vx, vy = np.meshgrid(x,y)

# Matrix for sample
S = np.zeros((nsamp,ninputs))
# Lower and upper bound vectors
lb = np.zeros((ninputs,1))
ub = np.ones((ninputs,1))
# List to store all results for all combinations
Results = []
Z_Imean = np.zeros((ncalc,ncalc))

fig = plt.figure(1)
# Looping through the 1st input until the second last
for i in range(ninputs-1):
    # Looping from the i+1 input until the last one
    for j in range(i+1,ninputs):
        # Constructing the sample space - testing set
        S = lhsamp(nsamp,ninputs,lb,ub)
        
        # Moving along the grid points
        for k in range(len(vx)):
            for l in range(len(vy)):
                tempX = vx[k,l]*np.ones((nsamp,1))
                tempY = vy[k,l]*np.ones((nsamp,1))
                # Getting the sample space as the testing set
                testing = S
                # Replace i and j columns with values from the grid
                testing[:,i] = tempX.reshape(nsamp,)
                testing[:,j] = tempY.reshape(nsamp,)
                np.savetxt('testfile',testing)
                subprocess.call('./filter')
                filter_out = np.loadtxt('filterout.dat')
                Z_mean = np.mean(filter_out[:,0])
                Z_Imean[k,l] = Z_mean
                Z_var = np.mean(filter_out[:,1])
                
                # Saving all combinations of results
                Results.append([i,j,vx[k,l],vy[k,l],Z_mean,Z_var])
        
        
        ax = fig.add_subplot(9,9,((9*i)+(i+j)-i))
        ax.contour(vx,vy,Z_Imean)
        plt.locator_params(nbins=5)

fig.tight_layout()

allResults = np.asarray(Results)
# np.savetxt(''.join([folder,'allResults_',observable,'.dat']),allResults)
np.savetxt('allResults.dat',allResults)

plt.show()