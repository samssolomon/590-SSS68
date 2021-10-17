
from keras.datasets import mnist
from keras.datasets import fashion_mnist
from keras.datasets import cifar10

from keras.utils import to_categorical

from keras import layers
from keras import models
from keras import optimizers

import matplotlib.pyplot as plt
import numpy as np


#----------------------------------------
# Set hyperparameters
#----------------------------------------

data = 'CIFAR-10' 	# Data can be MNIST, MNIST Fashion, or CIFAR-10

kernel_size = 3; 		# Kernel height and width defaults to 3, is otherwise 2 for CIFAR-10
pool_size = 2;			# Pool size defaults to 2
n_channels = 1;			# Number of channels defaults to 1 (for grayscale images), is 3 for CIFAR-10 (RGB images)

pixel_width = 28;		# Pixel width defaults to 28, is 32 for CIFAR-10
pixel_height = 28;		# Pixel height defaults to 28, is 32 for CIFAR-10
train_samples = 60000;	# Number of samples in training sets, defaults to 60,000 but is 50,000 for CIFAR-10
test_samples = 10000;	# Number of samples in test set, defaults to 10,000

num_epochs = 10;		# Number of epochs for training		
batch_size = 64;		# Batch size for training

if(data == 'CIFAR-10'): # Settings for CIFAR-10
	kernel_size = 2;
	n_channels = 3;
	pixel_width = 32;
	pixel_height = 32;
	train_samples = 50000;



#----------------------------------------
# Load data
#----------------------------------------

if(data == 'MNIST'):
	(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

	train_images = train_images.reshape((train_samples,pixel_width,pixel_height,n_channels))
	train_images = train_images.astype('float32') / 255

	test_images = test_images.reshape((test_samples,pixel_width,pixel_height,n_channels))
	test_images = test_images.astype('float32') / 255

	train_labels = to_categorical(train_labels)
	test_labels = to_categorical(test_labels)

	class_names = ['0','1','2','3','4','5','6','7','8','9']

if(data == 'MNIST Fashion'):
	(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

	train_images = train_images.reshape((train_samples,pixel_width,pixel_height,n_channels))
	train_images = train_images.astype('float32') / 255

	test_images = test_images.reshape((test_samples,pixel_width,pixel_height,n_channels))
	test_images = test_images.astype('float32') / 255

	train_labels = to_categorical(train_labels)
	test_labels = to_categorical(test_labels)

	class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
	'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


if(data == 'CIFAR-10'):
	(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

	train_images = train_images.reshape((train_samples,pixel_width,pixel_height,n_channels))
	train_images = train_images.astype('float32') / 255

	test_images = test_images.reshape((test_samples,pixel_width,pixel_height,n_channels))
	test_images = test_images.astype('float32') / 255

	train_labels = to_categorical(train_labels)
	test_labels = to_categorical(test_labels)


#------------------------------
# Visualize images from dataset
#------------------------------

# First image

plt.figure()
plt.imshow(train_images[2])
plt.colorbar()
plt.grid(False)
plt.show()






#----------------------------------------
# Instantiate convolutional neural network
#----------------------------------------

model = models.Sequential()
model.add(layers.Conv2D(32,(kernel_size, kernel_size), activation = 'relu', input_shape=(pixel_width,pixel_height, n_channels)))
model.add(layers.MaxPooling2D((pool_size, pool_size)))
model.add(layers.Conv2D(64, (kernel_size, kernel_size), activation = 'relu'))
model.add(layers.MaxPooling2D((pool_size, pool_size)))
model.add(layers.Conv2D(64, (kernel_size, kernel_size), activation = 'relu'))

model.summary()

#----------------------------------------
# Add classifier on top of the convolutional neural network
#----------------------------------------

model.add(layers.Flatten())
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(10, activation = 'softmax'))

model.summary()

#----------------------------------------
# Train the convolutional neural network and classifier on the data
#----------------------------------------

model.compile(optimizer='rmsprop',
			  loss='categorical_crossentropy',
			  metrics=['accuracy'])

history = model.fit(train_images,
					train_labels,
					epochs = num_epochs,
					batch_size = batch_size,
					validation_split = 0.2)



#------------------------------
# Display curves of loss and accuracy during training
#------------------------------

import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

#------------------------------
# Evaluate model on test data
#------------------------------

test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test loss:", test_loss, "Test accuracy:", test_acc)
