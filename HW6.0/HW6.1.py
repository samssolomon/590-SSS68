
import matplotlib.pyplot as plt
import numpy as np

from keras import models
from keras import layers

#-----------DEFINE USER PARAMETERS

epoch_num = 20;     # Number of training epochs
batch_n=1000;       # Batch size for training
n_bottleneck=10;    # Number of neurons in bottleneck hidden layer

#----------GET MNIST DATASET
from keras.datasets import mnist
(X, Y), (test_images, test_labels) = mnist.load_data()

#-----------NORMALIZE
X=X/np.max(X) 


#-----------RESHAPE
X=X.reshape(60000,28*28);  



#----------BUILD DEEP FEED FORWARD AUTO-ENCODER
model = models.Sequential()
model.add(layers.Dense(100,  activation='relu', input_shape=(28 * 28,)))
model.add(layers.Dense(n_bottleneck, activation='relu'))
model.add(layers.Dense(100,  activation='relu'))
model.add(layers.Dense(28*28,  activation='relu'))

#----------------COMPILE AND FIT MODEL
model.compile(optimizer='rmsprop',
                loss='mean_squared_error')
model.summary()

history = model.fit(X,X, 
         epochs=epoch_num,
         batch_size=batch_n,
         validation_split=0.2)

model.save('Deep Feed Forward Auto-encoder.h5')


#------------------PLOT TRAINING/VALIDATION HISTORY
def report(history,title='',I_PLOT=True):

    if(I_PLOT):
        #PLOT HISTORY
        epochs = range(1, len(history.history['loss']) + 1)
        plt.figure()
        plt.plot(epochs, history.history['loss'], 'b', label='Training loss')
        plt.plot(epochs, history.history['val_loss'], 'r', label='Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and validation loss')
        plt.legend()
        # plt.show()

        plt.savefig('HISTORY-'+title+'.png')   # save the figure to file
        plt.close()

report(history,title="HW6.1")


#----------------PRINT EXAMPLES OF ENCODE AND DECODE TRAINING DATA
X1=model.predict(X)

#RESHAPE
X=X.reshape(60000,28,28);
X1=X1.reshape(60000,28,28);

# COMPARE ORIGINAL WITH RECONSTRUCTED IMAGES
f, ax = plt.subplots(3,2)
I1=int(np.random.uniform(0,X.shape[0],1)[0])
I2=int(np.random.uniform(0,X.shape[0],1)[0])
I3=int(np.random.uniform(0,X.shape[0],1)[0])
ax[0,0].imshow(X[I1])
ax[0,1].imshow(X1[I1])
ax[1,0].imshow(X[I2])
ax[1,1].imshow(X1[I2])
ax[2,0].imshow(X[I3])
ax[2,1].imshow(X1[I3])
plt.show()

#----------------EVALUATE ON TEST DATA FROM MNIST

#-----------NORMALIZE
test_images=test_images/np.max(test_images) 

#-----------RESHAPE
test_images=test_images.reshape(10000,28*28);

results_test = model.evaluate(test_images,test_images)
print("Model mean squared error with test data from MNIST:", results_test)





#----------------ANOMALY DETECTION
#--------------Set reconstruction loss threshold (using MSE)

# Get train MSE loss for each sample
X=X.reshape(60000,28*28)
X1=X1.reshape(60000,28*28)
train_mse_loss = np.mean((X1-X)**2,axis=1)

# Plot MSE loss on MNIST training set
plt.hist(train_mse_loss, bins=50)
plt.xlabel("MNIST training set MSE loss")
plt.ylabel("No of samples")
plt.show()

# Get reconstruction loss threshold
threshold = np.mean(train_mse_loss) + 3*np.std(train_mse_loss)
print("Reconstruction error threshold: ", threshold)

# Detect all the samples from Fashion MNIST which are anomalies
anomalies1 = train_mse_loss > threshold

# Print summary
print("Number of anomaly samples in MNIST training set: ", np.sum(anomalies1))
print("Percent of samples designated as anomalies in MNIST training set: ", 100*np.sum(anomalies1)/X.shape[0],"%")


# Load fashion_mnist dataset (images of which are anomalies)
from keras.datasets import fashion_mnist 
(f_X, f_Y), (f_test_images, f_test_labels) = fashion_mnist.load_data()

#-----------NORMALIZE
f_X=f_X/np.max(f_X) 

#-----------RESHAPE
f_X=f_X.reshape(60000,28*28);  

#-----------GENERATE AUTOENCODER PREDICTIONS FOR Fashion_MNIST
f_X_pred = model.predict(f_X)

#RESHAPE
f_X=f_X.reshape(60000,28,28);
f_X_pred=f_X_pred.reshape(60000,28,28);

# COMPARE ORIGINAL WITH RECONSTRUCTED IMAGES
g, bx = plt.subplots(3,2)
I4=int(np.random.uniform(0,f_X.shape[0],1)[0])
I5=int(np.random.uniform(0,f_X.shape[0],1)[0])
I6=int(np.random.uniform(0,f_X.shape[0],1)[0])
bx[0,0].imshow(f_X[I4])
bx[0,1].imshow(f_X_pred[I4])
bx[1,0].imshow(f_X[I5])
bx[1,1].imshow(f_X_pred[I5])
bx[2,0].imshow(f_X[I6])
bx[2,1].imshow(f_X_pred[I6])
plt.show()

# Find MSE for each sample
f_X=f_X.reshape(60000,28*28)
f_X_pred=f_X_pred.reshape(60000,28*28)
f_mnist_mse_loss = np.mean((f_X_pred - f_X)**2, axis=1)

# Plot MSE for Fashion MNIST dataset
plt.hist(f_mnist_mse_loss, bins=50)
plt.xlabel("Fashion MNIST training set MSE loss")
plt.ylabel("No of samples")
plt.show()

# Detect all the samples from Fashion MNIST which are anomalies
anomalies2 = f_mnist_mse_loss > threshold

# Print summary
print("Number of anomaly samples in Fashion MNIST training set: ", np.sum(anomalies2))
print("Percent of samples designated as anomalies in Fashion MNIST training set: ", 100*np.sum(anomalies2)/f_X.shape[0],"%")