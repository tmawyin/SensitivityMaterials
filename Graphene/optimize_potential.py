''' Use surrogate models to optimize input parameters of the graphene interatomic potential '''

import numpy as np
import subprocess
from lammps import lammps

# Loading the file - trimmed data
file = np.loadtxt('Data/allData.dat')
# File to save all data at the end
Full_file = open('Full_File.dat','a')

# Keeping the second element of the matrix
toKeep = file[1,:]
toKeep_In = toKeep[0:10]
toKeep_Out = toKeep[10:14]
data = np.delete(file,1,0)

# Variables from the data
nRows, nCols = data.shape
nPts = nRows
nOutputs = 4
nInputs = nCols-nOutputs

# Break data in input and outputs
data_In = data[:,0:10]
data_Out = data[:,10:14]

# Calculating upper and lower bounds
lb = np.min(data,0)
ub = np.max(data,0)

# Normalizing input data
data_In_norm = np.zeros((nPts,nInputs))
for i in range(nInputs):
    data_In_norm[:,i] = (data_In[:,i]-lb[i])/(ub[i]-lb[i])

# Initial guess
# guess = np.mean(data_In_norm,0)
# guess = np.ones((1,14))*0.5
guess = np.random.uniform(0,1,size=(1,14))

# Calculating the least squares error
lse = np.sum(((toKeep_Out-data_Out)**2)/toKeep_Out,axis=1)
lse = lse.reshape(len(lse),1)

# Combining the normalized input and the LSE from outputs
toFile_InOut = np.hstack((data_In_norm,lse))

# Saving to full file
Full_file.write('Full Data File\n')
np.savetxt(Full_file, toKeep_In , newline=" ")
Full_file.write('\n')
np.savetxt(Full_file, toKeep_Out , newline=" ")
Full_file.write('\n------------------\n')


# Creating the input file
fp = open('inputFile.dat','w')
fp.write('%d   ! No. of training points\n' %nRows)
fp.write('%d   ! No. of input variables\n' %nInputs)
fp.write('0.1  ! Regularization parameter\n')
fp.write('0    ! Objective function flag\n')
fp.write('0    ! print flag\n')
fp.write('100000 ! Maximum number of approximate function evaluations\n')
fp.write('12410  ! Seed for RNG\n')
np.savetxt(fp, np.zeros((1,10)), newline=" ")
fp.write('    ! Lower bound\n')
np.savetxt(fp, np.ones((1,10)), newline=" ")
fp.write('    ! Upper bound\n')
np.savetxt(fp, guess[0:10], newline=" ")
fp.write('    ! Initial guess\n')
np.savetxt(fp, toFile_InOut)
fp.close()

# Running the surrogate model
subprocess.call('./saei',stdin=open('inputFile.dat'))

# Opening results and "un-normalizing" them
Rslt = np.loadtxt('optim.dat')
Results_norm = np.zeros((nInputs))
Results_surrogate = np.zeros((nInputs))
for i in range(nInputs):
	Results_surrogate[i] = Rslt[i]
	Results_norm[i] = Rslt[i]*(ub[i]-lb[i])+lb[i]


# Copy results to a new potential
# Opening file to read lines and modify them
Pot_file = open('airebo_part1.txt', 'r')
lines = Pot_file.readlines()
lines[19] = '%.16f    Q_CC\n' %Results_norm[0]
lines[22] = '%.16f    alpha_CC\n' %Results_norm[1]
lines[37] = '%.16f    Beta_CC1\n' %Results_norm[2]
lines[38] = '%.16f    Beta_CC2\n' %Results_norm[3]
lines[39] = '%.16f    Beta_CC3\n' %Results_norm[4]
lines[49] = '%.16f    rcLJmin_CC\n' %Results_norm[5]
lines[52] = '%.16f    rcLJmax_CC\n' %Results_norm[6]
lines[61] = '%.16f    epsilon_CC\n' %Results_norm[7]
lines[64] = '%.16f    sigma_CC\n' %Results_norm[8]
lines[67] = '%.16f    epsilonT_CCCC\n' %Results_norm[9]
Pot_file.close()

# Writing the lines to file
Pot_file = open('airebo_part1.txt', 'w')
Pot_file.writelines(lines)
Pot_file.close()

# Create the potential file by running the concatinate.sh script
subprocess.call('./ConcatFiles.sh')

# Run Elastic Constants and Evacancy
Rslt_lmp_vars = np.zeros((1,nOutputs))
lmp_EC = lammps()
lmp_EC.file('graphene_elastic.in')
# Capturing variables C11, C12, and Ecoh
Rslt_lmp_vars[0,0] = lmp_EC.extract_variable('C11','all',0)
Rslt_lmp_vars[0,1] = lmp_EC.extract_variable('C12','all',0)
Rslt_lmp_vars[0,2] = lmp_EC.extract_variable('Ecoh','all',0)
lmp_EC.close()

lmp_Ev = lammps()
lmp_Ev.file('graphene_vacancy.in')
Rslt_lmp_vars[0,3] = lmp_Ev.extract_variable('Ev','all',0)
lmp_Ev.close()

# Calculate the error of the observables
lse_Rsl = np.sum(((toKeep_Out-Rslt_lmp_vars)**2)/toKeep_Out,axis=1)
lse_Rsl = lse_Rsl.reshape(len(lse_Rsl),1)

# Calculate the error on the input
lse_Inputs = np.sum(((toKeep_In-Results_norm)**2)/toKeep_In,axis=0)

# Saving to full file
np.savetxt(Full_file, Results_norm , newline=" ")
Full_file.write('\n')
np.savetxt(Full_file, Rslt_lmp_vars , newline=" ")
Full_file.write('\n')
np.savetxt(Full_file, lse_Rsl , newline=" ")
Full_file.write('\n%.10f'%(float(lse_Inputs)))
Full_file.write('\n------------------\n')
Full_file.close()


''' For Comparison Purposes: '''
''' Testing using the surrogate '''
# Creating the parameters file
fp = open('gppar.dat','w')
fp.write('%d\n' %(nInputs))	# Number of inputs
fp.write('%d\n' %(nPts))	# Number of training points
fp.write('1\n')				# Number of testing points
fp.write('0\n')
fp.write('2\n')
fp.write('trainfile\n')
fp.write('testfile\n')
fp.close()

# Creating the results file
Rslt_GPM_mean = np.zeros((1,nOutputs))
Rslt_GPM_var = np.zeros((1,nOutputs))
print Results_surrogate.shape
# Saving the testing file
np.savetxt('testfile',Results_surrogate.reshape(1,nInputs))

# Calculating the GPM for all the observables
for i in range(nOutputs):
	training = np.hstack((data_In_norm,data_Out[:,i:i+1]))
	np.savetxt('trainfile',training)
	# Calling the GPM surrogate
	subprocess.call('./filter')
	filter_out = np.loadtxt('filterout.dat')
	Rslt_GPM_mean[0,i] = filter_out[0]
	Rslt_GPM_var[0,i] = filter_out[1]


# Printing results
print "The new output error is:", lse_Rsl
print "The new input error is:", lse_Inputs

print "The real values are:"
print toKeep_Out
print "The result from MD values are:"
print Rslt_lmp_vars
print "The result from GPM values are:"
print Rslt_GPM_mean