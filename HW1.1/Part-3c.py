# Set working directory
import os
os.chdir('/home/sam590/590-CODES/DATA')

# Read data
import json
import numpy as np
weight = open('weight.json',)
data = json.load(weight)

# Extract x and y from json, convert to numpy arrays
x = np.array(data['y'])
y = np.array(data['is_adult'])

# Normalize data
x_mean = np.mean(x)
x_std = np.std(x)
x_norm = (x - x_mean)/x_std


# Partition data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_norm, y, test_size = 0.2, random_state = 123)

# Define model
def model(x,p):
     global model_type
     if(model_type=="linear"): return p[0]*x+p[1]  
     if(model_type=="logistic"): return  p[0]+p[1]*(1.0/(1.0+np.exp(-(x-p[2])/(p[3]+0.00001))))

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

res = minimize(loss, po, method='Nelder-Mead', tol=1e-15)
popt = res.x
print("Optimal parameters are:",popt)

# Generate predictions and unnormalize data

x_unnorm = x_std * x_train + x_mean
ypred_unnorm = model(x_train,popt)
ypred = y_std * ypred_unnorm + y_mean

logcurve_x = np.array(list(range(3,100)))
logcurve_x_norm = (logcurve_x - x_mean)/x_std
logcurve_pred_norm = model(logcurve_x_norm,popt)
logcurve_pred = y_std * logcurve_pred_norm + y_mean

# Plot 
import matplotlib.pyplot as plt
plt.figure() # Initialize figure
FS=18   # Font size
plt.scatter(x,y)
plt.plot(logcurve_x,logcurve_pred, color='red')
plt.xlabel('age (years)', fontsize=FS)
plt.ylabel('weight(lb)', fontsize=FS)
plt.title('Logistic Regression of Adulthood on Weight', fontsize=FS)
plt.show()


