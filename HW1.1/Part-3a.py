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

# Select only minors (age under 18) for linear regression
x_under_18 = x[x<18]
y_under_18 = y[x<18]

# Normalize data
x_mean = np.mean(x_under_18)
x_std = np.std(x_under_18)
x_norm = (x_under_18 - x_mean)/x_std

y_mean = np.mean(y)
y_std = np.std(y)
y_norm = (y_under_18 - y_mean)/y_std

# Partition data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_norm, y_norm, test_size = 0.2, random_state = 123)

# Define model
def model(x,p):
     global model_type
     if(model_type=="linear"): return p[0]*x+p[1]  
     if(model_type=="logistic"): return  p[0]+p[1]*(1.0/(1.0+np.exp(-(x-p[2])/(p[3]+0.00001))))

# Set model type
model_type = "linear"

# Define loss function for model
def loss(p):
	yp = model(x_train,p) 
	MSE = (y_train - yp)**2))/(y_train.size)
	loss = MSE
	return loss

# Call optimizer from SciPy and run on training data
from scipy.optimize import minimize
NFIT = 2 # number of parameters
np.random.seed(42)
po = np.random.uniform(0.5,1.,size=NFIT) # Random initial guess for fitting parameters

res = minimize(loss, po, method='Nelder-Mead', tol=1e-15)
popt = res.x
print("Optimal parameters are:",popt)

# Generate predictions and unnormalize data

x_unnorm = x_std * x_train + x_mean
ypred_unnorm = model(x_train,popt)
ypred = y_std * ypred_unnorm + y_mean

# Plot 
import matplotlib.pyplot as plt

m, b = np.polyfit(x_unnorm,ypred,1) 
xline = np.array(list(range(3,28)))

plt.figure() # Initialize figure
FS=18   # Font size
plt.scatter(x,y)
plt.plot(xline,m*xline+b, color='red')
plt.xlabel('age (years)', fontsize=FS)
plt.ylabel('weight(lb)', fontsize=FS)
plt.ylim([25,225])
plt.title('Linear Regression of Weight on Age', fontsize=FS)
ax.text(40, 150, r"$y = f(x | m,b) $", color="r", fontsize=20)
plt.show()
