import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from keras import models
from keras import layers



#GET DATASET
# from keras.datasets import mnist
# (X, Y), (test_images, test_labels) = mnist.load_data()

from keras.datasets import fashion_mnist 
(X, Y), (test_images, test_labels) = fashion_mnist.load_data()

#NORMALIZE
X=X/np.max(X) 

#INJECT NOISE
X2=X+1*np.random.uniform(0,1,X.shape)

#RESHAPE
X=X.reshape(60000,28*28);  
X2=X2.reshape(60000,28*28);

#MODEL
n_bottleneck=200

#SHALLOW 
model = models.Sequential()
model.add(layers.Dense(n_bottleneck, activation='relu', input_shape=(28 * 28,)))
model.add(layers.Dense(28*28,  activation='relu'))

#DEEPER
# model = models.Sequential()
# model.add(layers.Dense(400,  activation='relu', input_shape=(28 * 28,)))
# model.add(layers.Dense(n_bottleneck, activation='relu'))
# model.add(layers.Dense(400,  activation='relu'))
# model.add(layers.Dense(28*28,  activation='relu'))

#COMPILE+FIT
model.compile(optimizer='rmsprop',
                loss='mean_squared_error')
model.summary()
model.fit(X2, X, epochs=20, batch_size=1000,validation_split=0.2)
X1=model.predict(X)

#RESHAPE FOR PLOTTING
X=X.reshape(60000,28,28);  
X1=X1.reshape(60000,28,28); 
X2=X2.reshape(60000,28,28);  

#COMPARE ORIGINAL 
f, ax = plt.subplots(6,1)
I1=int(np.random.uniform(0,X.shape[0],1)[0])
I2=int(np.random.uniform(0,X.shape[0],1)[0])
ax[0].imshow(X[I1])
ax[1].imshow(X2[I1])
ax[2].imshow(X1[I1])
ax[3].imshow(X[I2])
ax[4].imshow(X2[I2])
ax[5].imshow(X1[I2])
plt.show()

#XTRA CODE 

#INJECT NOISE
# f, ax = plt.subplots(2,1)
# I1=1;
# ax[0].imshow(X[I1])
# X2=X+1*np.random.uniform(0,1,X.shape)
# ax[1].imshow(X2[I1])
# plt.show()
