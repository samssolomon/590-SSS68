import numpy as np # Load packages
import matplotlib
import matplotlib.pyplot as plt
import json
from   scipy.optimize import minimize

#USER PARAMETERS
IPLOT=True #To make interactive plots available
INPUT_FILE='weight.json' #To define the dataset
FILE_TYPE="json"
DATA_KEYS=['x','is_adult','y']


#UNCOMMENT FOR VARIOUS MODEL CHOICES (ONE AT A TIME)
model_type="logistic"; NFIT=4; xcol=1; ycol=2;
# model_type="linear";   NFIT=2; xcol=1; ycol=2; 
# model_type="logistic";   NFIT=4; xcol=2; ycol=0;


#READ FILE
with open(INPUT_FILE) as f:
	my_input = json.load(f)  #read into dictionary


#CONVERT INPUT INTO ONE LARGE MATRIX (SIMILAR TO PANDAS DF)
X=[];
for key in my_input.keys():
	if(key in DATA_KEYS): X.append(my_input[key])

#MAKE ROWS=SAMPLE DIMENSION (TRANSPOSE)
X=np.transpose(np.array(X))

#SELECT COLUMNS FOR TRAINING 
x=X[:,xcol];  y=X[:,ycol]

#EXTRACT AGE<18
if(model_type=="linear"):
	y=y[x[:]<18]; x=x[x[:]<18]; 

#COMPUTE BEFORE PARTITION AND SAVE FOR LATER
XMEAN=np.mean(x); XSTD=np.std(x)
YMEAN=np.mean(y); YSTD=np.std(y)

#NORMALIZE
x=(x-XMEAN)/XSTD;  y=(y-YMEAN)/YSTD; 

#PARTITION
f_train=0.8; f_val=0.2
rand_indices = np.random.permutation(x.shape[0])
CUT1=int(f_train*x.shape[0]); 
train_idx,  val_idx = rand_indices[:CUT1], rand_indices[CUT1:]
xt=x[train_idx]; yt=y[train_idx]; xv=x[val_idx];   yv=y[val_idx]

#MODEL
def model(x,p):
	if(model_type=="linear"):   return  p[0]*x+p[1]  
	if(model_type=="logistic"): return  p[0]+p[1]*(1.0/(1.0+np.exp(-(x-p[2])/(p[3]+0.00001))))

#SAVE HISTORY FOR PLOTTING AT THE END
iteration=0; iterations=[]; loss_train=[];  loss_val=[]

#LOSS FUNCTION
def loss(p,index_2_use):
	global iterations,loss_train,loss_val,iteration

	#TRAINING LOSS
	yp=model(x[index_2_use],p) #model predictions for given parameterization p
	training_loss=(np.mean((yp-y[index_2_use])**2.0))  #MSE

	return training_loss


#TRAIN MODEL USING OPTIMIZER THAT WORKS WITH GRADIENT DESCENT
def optimizer(f,xi, algo):
	global epoch, loss_train, loss_val
	# algo can be 'GD' for gradient descent or 'MOM' for gradient descent with momentum

	#PARAM
	dx=0.001							#STEP SIZE FOR FINITE DIFFERENCE
	LR=0.05								#LEARNING RATE
	iteration=1 	 					#INITIAL ITERATION COUNTER
	maxiter=100000						#MAX NUMBER OF ITERATIONS
	tol=10**-10							#EXIT AFTER CHANGE IN F IS LESS THAN THIS 
	alpha=0.1							#EXPONENTIAL DECAY FACTOR FOR MOMENTUM
	epoch=0								#EPOCH
	NDIM=len(xi)						#NUMBER OF DIMENSIONS OF OPTIMIZATION PROBLEM

	last_change=0 						#SET UP VARIABLE FOR MOMENTUM ALGORITHM

	if(PARADIGM=='stochastic'):
		LR=0.005; maxiter=30000			#SET DIFFERENT LEARNING RATE AND MAX NUMBER OF ITERATIONS FOR STOCHASTIC PARADIGM


	while(iteration<=maxiter):

		#USE TRAINING PARADIGM TO SELECT DATA FOR TRAINING
		if(PARADIGM=='batch'): # Do the following for the batch paradigm
			if(iteration==1): epoch+=1; index_2_use=train_idx # Select all data from the training data and update epoch number on the first pass
			if(iteration>1): epoch+=1 # Update epoch number on subsequent iterations
		elif(PARADIGM=='mini'): # Do the following for the mini-batch paradigm
			mini_size = int(train_idx.shape[0]/2) # Define mini-batch size as half the training sample
			mini1_idx = train_idx[0:mini_size] # Define indices for first mini-batch
			mini2_idx = train_idx[mini_size:len(train_idx)] # Define indices for second mini-batch
			if(iteration % 2 == 1): index_2_use = mini1_idx; epoch +=1 # If odd iteration, train on first mini-batch and update epoch number
			if(iteration % 2 == 0): index_2_use = mini2_idx # If even iteration, train on second mini-batch
		elif(PARADIGM=='stochastic'): # Do the following for the stochastic paradigm
			if(iteration==1): epoch+=1 # Update epoch number on first pass
			index_2_use = (iteration % len(train_idx)) # Train on individual data point for each pass
			if(iteration>1): epoch+=1 # Update epoch number on subsequent iterations
		else:
			print("PARADIGM NOT CODED") # Throw error message if training paradigm not coded

		#NUMERICALLY COMPUTE GRADIENT OF LOSS FUNCTION
		df_dx=np.zeros(NDIM) # Initiatilize gradient vector
		for i in range(0,NDIM): # Set up loop for number of dimensions

			dX=np.zeros(NDIM); # Initialize step array
			dX[i]=dx; # Define step of length dx along ith dimension
			xm1=xi-dX; # Step backward by length dx along ith dimension
			xp1=xi+dX; # Step forward by length dx along ith dimension

			# Find gradient of loss function in ith dimension numerically through central finite difference
			grad_i=(f(xp1,index_2_use) - f(xm1,index_2_use))/dx/2 

			# Update gradient vector for each dimension
			df_dx[i]=grad_i

		#OPTIMIZER TAKES A STEP ALONG GRADIENT VECTOR
		if(algo=="GD"): 
			change=LR*df_dx # Define change in guess for parameters by multiplying learning rate by gradient vector
			xip1=xi-change # Update guess for parameters 
		xip1=xi-LR*df_dx # For gradient descent, update guess for parameters by multiplying learning rate by gradient vector 
		if(algo=="MOM"): 
			change=LR*df_dx + alpha*last_change # Define change in guess for parameters with term for last update and exponential decay factor for momentum
			xip1=xi-change # Update guess for parameters
			last_change=change # Update last change variable
		#if(algo="RMSprop"):
		#if(algo='ADAM'):

		if(iteration%10==0): # On every tenth iteration
			df=np.mean(np.absolute(f(xip1,index_2_use)-f(xi,index_2_use)))

			yp=model(xt,xi) # Model predictions for given parameterization p with training data
			training_loss=(np.mean((yp-yt)**2.0)) # Find MSE for training data with given parameterization p

			#VALIDATION LOSS
			yp=model(xv,xi) # Model predictions for given parameterization p with validation data
			validation_loss=(np.mean((yp-yv)**2.0)) # Find MSE for validation data with given parameterization p
	
			#RECORD FOR PLOTING
			loss_train.append(training_loss); loss_val.append(validation_loss)
			iterations.append(iteration);
		
			#WRITE TO SCREEN
			print(iteration," ",epoch," ",training_loss," ",validation_loss," ",xi)

			#BREAK LOOP IF
			if(df<tol):
				print("STOPPING CRITERION MET (STOPPING TRAINING)")
				break

		#UPDATE FOR NEXT ITERATION OF LOOP
		xi=xip1 # Update guess for parameters of model for next pass through
		iteration=iteration+1 # Update iteration at end of loop

	return xi



#DEFINE TRAINING PARADIGM
#PARADIGM="batch"
PARADIGM="mini"
#PARADIGM="stochastic"

#MAKE INITIAL GUESS FOR PARAMETERS
p0=np.random.uniform(0.5,1.,size=NFIT)

#TRAIN MODEL USING OPTIMIZER THAT WORKS WITH GRADIENT DESCENT
popt = optimizer(loss, p0, algo = 'MOM')

#PRINT OPTIMAL PARAMETERS
print("OPTIMAL PARAM:",popt)


#PREDICTIONS
xm=np.array(sorted(xt))
yp=np.array(model(xm,popt))

#UN-NORMALIZE
def unnorm_x(x): 
	return XSTD*x+XMEAN  
def unnorm_y(y): 
	return YSTD*y+YMEAN 

#FUNCTION PLOTS
if(IPLOT):
	fig, ax = plt.subplots()
	ax.plot(unnorm_x(xt), unnorm_y(yt), 'o', label='Training set')
	ax.plot(unnorm_x(xv), unnorm_y(yv), 'x', label='Validation set')
	ax.plot(unnorm_x(xm),unnorm_y(yp), '-', label='Model')
	plt.xlabel('x', fontsize=18)
	plt.ylabel('y', fontsize=18)
	plt.legend()
	plt.show()

#PARITY PLOTS
if(IPLOT):
	fig, ax = plt.subplots()
	ax.plot(model(xt,popt), yt, 'o', label='Training set')
	ax.plot(model(xv,popt), yv, 'o', label='Validation set')
	plt.xlabel('y predicted', fontsize=18)
	plt.ylabel('y data', fontsize=18)
	plt.legend()
	plt.show()

#MONITOR TRAINING AND VALIDATION LOSS  
if(IPLOT):
	fig, ax = plt.subplots()
	ax.plot(iterations, loss_train, 'o', label='Training loss')
	ax.plot(iterations, loss_val, 'o', label='Validation loss')
	plt.xlabel('optimizer iterations', fontsize=18)
	plt.ylabel('loss', fontsize=18)
	plt.legend()
	plt.show()
