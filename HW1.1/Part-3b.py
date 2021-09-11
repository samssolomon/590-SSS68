# Set working directory
import os
os.chdir('/home/sam590/590-CODES/DATA')

# Read data
import json
import numpy as np
weight = open('weight.json',)
data = json.load(weight)

# Extract x and y from json, convert to numpy arrays
x = np.array(data['x'])
y = np.array(data['y'])

# Normalize data
x_mean = np.mean(x)
x_std = np.std(x)
x_norm = (x - x_mean)/x_std

y_mean = np.mean(y)
y_std = np.std(y)
y_norm = (y - y_mean)/y_std

# Partition data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_norm, y_norm, test_size = 0.2, random_state = 123)

# Define model
def model(x,p):
     global model_type
     if(model_type=="linear"): return p[0]*x+p[1]  
     if(model_type=="logistic"): return  p[0]+p[1]*(1.0/(1.0+np.exp(-(x-p[2])/(p[3]+0.00001))))

# Set model type
model_type = "logistic"

# Define loss function
def loss(p):
	yp = model(x_train,p) 
	MSE = np.mean((y_train - yp)**2)
	loss = MSE
	return loss

# Call optimizer from SciPy and run on training data
from scipy.optimize import minimize
NFIT = 4 # number of parameters
np.random.seed(43)
po = np.random.uniform(0.5,1.,size=NFIT) # Random initial guess for fitting parameters

res = minimize(loss, po, method='BFGS', tol=1e-15)
popt = res.x
print("Optimal parameters are:",popt)

#Save history for training loss at the end
iterations=[]; loss_train=[];  loss_val=[]
iteration=0
def loss(p):
	global iteration,iterations,loss_train,loss_val
	yp = model(x_train,p)
	training_loss = (y_train - yp)**2/(y_train.size)
	validation_loss = (y_test - yp)**2/(y_test.size)
	loss_train.append(training_loss)
	loss_val.append(validation_loss)
	iterations.append(iteration)
	iteration+=1

# Generate predictions and unnormalize data

ypred_norm = model(x_norm,popt)
ypred = y_std*ypred_norm + y_mean

logcurve_x = np.array(list(range(3,100)))
logcurve_x_norm = (logcurve_x - x_mean)/x_std
logcurve_pred_norm = model(logcurve_x_norm,popt)
logcurve_pred = y_std * logcurve_pred_norm + y_mean

# Plot 
import matplotlib.pyplot as plt
plt.figure() # Initialize figure
FS=18   # Font size
plt.scatter(x,y)
plt.plot(x,ypred,'-', color='red')
plt.xlabel('age (years)', fontsize=FS)
plt.ylabel('weight(lb)', fontsize=FS)
plt.ylim([25,225])
plt.title('Logistic Regression of Weight on Age', fontsize=FS)
plt.show()


