
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras

from keras import models
from keras import layers
from tensorflow.keras import datasets, layers, models, losses, Model
from random import randint

#-----------DEFINE USER PARAMETERS

epoch_num = 20;     # Number of training epochs
batch_n=1000;       # Batch size for training

#----------GET CIFAR-10 DATASET
(x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()

#-----------RESCALE
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

#-----------RESHAPE
x_train = np.reshape(x_train, (len(x_train), 32, 32, 3))
x_test = np.reshape(x_test, (len(x_test), 32, 32, 3))

#----------BUILD CONVOLUTIONAL AUTO-ENCODER

input_img = keras.Input(shape=(32, 32, 3))

x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(16, (3, 3), activation='relu',padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = keras.Model(input_img, decoded)

#----------------COMPILE AND FIT MODEL
autoencoder.compile(optimizer='rmsprop',
                loss='mean_squared_error')
autoencoder.summary()

history = autoencoder.fit(x_train,x_train, 
         epochs=epoch_num,
         batch_size=batch_n,
         validation_split=0.2)

autoencoder.save('Convolutional Auto-encoder_CIFAR.h5')

#------------------PLOT TRAINING/VALIDATION HISTORY
def report(history,title='',I_PLOT=True):

    if(I_PLOT):
        #PLOT HISTORY
        epochs = range(1, len(history.history['loss']) + 1)
        plt.figure()
        plt.plot(epochs, history.history['loss'], 'b', label='Training loss')
        plt.plot(epochs, history.history['val_loss'], 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        # plt.show()

        plt.savefig('HISTORY-'+title+'.png')   # save the figure to file
        plt.close()

report(history,title="HW6.3")

#----------------PRINT EXAMPLES OF ENCODE AND DECODE TRAINING DATA
x1=autoencoder.predict(x_train)

#RESHAPE
x_train=x_train.reshape(50000,32,32,3);
x1=x1.reshape(50000,32,32,3);

# COMPARE ORIGINAL WITH RECONSTRUCTED IMAGES
f, ax = plt.subplots(3,2)
I1=int(np.random.uniform(0,x_train.shape[0],1)[0])
I2=int(np.random.uniform(0,x_train.shape[0],1)[0])
I3=int(np.random.uniform(0,x_train.shape[0],1)[0])
ax[0,0].imshow(x_train[I1])
ax[0,1].imshow(x1[I1])
ax[1,0].imshow(x_train[I2])
ax[1,1].imshow(x1[I2])
ax[2,0].imshow(x_train[I3])
ax[2,1].imshow(x1[I3])
plt.show()

#----------------EVALUATE ON TEST DATA FROM MNIST

results_test = autoencoder.evaluate(x_test,x_test)
print("Model mean squared error with test data from CIFAR-10:", results_test)





#----------------ANOMALY DETECTION
#--------------Set reconstruction loss threshold (using MSE)

# Get train MSE loss for each sample
x_train=x_train.reshape(50000,32*32*3)
x1=x1.reshape(50000,32*32*3)
train_mse_loss = np.mean((x1-x_train)**2,axis=1)

# Plot MSE loss on MNIST training set
plt.hist(train_mse_loss, bins=50)
plt.xlabel("CIFAR-10 training set MSE loss")
plt.ylabel("No of samples")
plt.show()

# Get reconstruction loss threshold
threshold = np.mean(train_mse_loss) + 3*np.std(train_mse_loss)
print("Reconstruction error threshold: ", threshold)

# Detect all the samples from Fashion MNIST which are anomalies
anomalies1 = train_mse_loss > threshold

# Print summary
print("Number of anomaly samples in CIFAR-10 training set: ", np.sum(anomalies1))
print("Percent of samples designated as anomalies in CIFAR-10 training set: ", 100*np.sum(anomalies1)/x_train.shape[0],"%")


# Load CIFAR-100 dataset (images of which are anomalies)
(f_x_train, f_y_train), (f_x_test, f_y_train) = keras.datasets.cifar100.load_data()

#-----------REMOVE TRUCKS FROM DATASET
trucks = (f_y_train==[93])
truck_indices = np.where(trucks)[0]
f_x_train = np.delete(f_x_train, truck_indices,axis=0)

#-----------RESCALE
f_x_train = f_x_train.astype('float32') / 255.

#-----------RESHAPE
f_x_train = np.reshape(f_x_train, (len(f_x_train), 32, 32, 3))

#-----------GENERATE AUTOENCODER PREDICTIONS FOR Fashion_MNIST
f_x_pred = autoencoder.predict(f_x_train)

#-----------RESHAPE FOR PLOTS
f_x_train=f_x_train.reshape(49900,32,32,3);
f_x_pred=f_x_pred.reshape(49900,32,32,3);

# COMPARE ORIGINAL WITH RECONSTRUCTED IMAGES
g, bx = plt.subplots(3,2)
I4=int(np.random.uniform(0,f_x_train.shape[0],1)[0])
I5=int(np.random.uniform(0,f_x_train.shape[0],1)[0])
I6=int(np.random.uniform(0,f_x_train.shape[0],1)[0])
bx[0,0].imshow(f_x_train[I4])
bx[0,1].imshow(f_x_pred[I4])
bx[1,0].imshow(f_x_train[I5])
bx[1,1].imshow(f_x_pred[I5])
bx[2,0].imshow(f_x_train[I6])
bx[2,1].imshow(f_x_pred[I6])
plt.show()

# Find MSE for each sample
f_x_train=f_x_train.reshape(49900,32*32*3)
f_x_pred=f_x_pred.reshape(49900,32*32*3)
cifar100_mse_loss = np.mean((f_x_pred - f_x_train)**2, axis=1)

# Plot MSE for Fashion MNIST dataset
plt.hist(cifar100_mse_loss, bins=50)
plt.xlabel("CIFAR-100 training set MSE loss")
plt.ylabel("No of samples")
plt.show()

# Detect all the samples from Fashion MNIST which are anomalies
anomalies2 = cifar100_mse_loss > threshold

# Print summary
print("Number of anomaly samples in CIFAR-100 training set: ", np.sum(anomalies2))
print("Percent of samples designated as anomalies in CIFAR-100 training set: ", 100*np.sum(anomalies2)/f_x_train.shape[0],"%")