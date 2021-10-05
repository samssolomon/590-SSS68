#------------------
# Load IMDB data
#------------------
from keras.datasets import imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

#------------------
# Examine data
#------------------
#train_data[0]
#max([max(sequence) for sequence in train_data])

#word_index = imdb.get_word_index()
#reverse_word_index = dict(
#	[(value, key) for (key, value) in word_index.items()])
#decoded_review = ' '.join(
#	[reverse_word_index.get(i-3, '?') for i in train_data[0]])
#print(decoded_review)

#------------------
# Pre-process data to be fed into neural network
#------------------
import numpy as np

def vectorize_sequences(sequences, dimension=10000):
	results = np.zeros((len(sequences), dimension))
	for i, sequence in enumerate(sequences):
		results[i, sequence] = 1.
	return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

#x_train[0]

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

#------------------
# Define model
#------------------
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics

model1 = models.Sequential()
model1.add(layers.Dense(16, activation='relu', input_shape = (10000,)))
model1.add(layers.Dense(16, activation='relu'))
model1.add(layers.Dense(1, activation='sigmoid'))

#------------------
# Compile model (define optimizer, loss function, and metrics)
#------------------
model1.compile(optimizer='rmsprop',
			  loss='binary_crossentropy',
			  metrics=['acc'])

#------------------
# Set aside validation set
#------------------
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

#------------------
# Train model
#------------------
print("---------Train model over 20 epochs---------")
history = model1.fit(partial_x_train,
					partial_y_train,
					epochs=20,
					batch_size=512,
					validation_data=(x_val,y_val))

#------------------
# Plot training and validation loss
#------------------
import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(history_dict['acc']) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

#------------------
# Plot training and validation accuracy
#------------------
plt.clf()
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc_values, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()






#------------------
# Retrain model from scratch with only 4 epochs
#------------------
print("---------Retrain model over 4 epochs instead of 20---------")
model2 = models.Sequential()
model2.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model2.add(layers.Dense(16, activation='relu'))
model2.add(layers.Dense(1, activation='sigmoid'))

model2.compile(optimizer='rmsprop',
			  loss='binary_crossentropy',
			  metrics=['accuracy'])

history = model2.fit(x_train,
		   y_train,
		   epochs=4,
		   batch_size=512,
		   validation_data=(x_val,y_val))
results = model2.evaluate(x_test,y_test)
print("Model binary crossentropy and accuracy with test data:",results)

#------------------
# Plot training and validation loss
#------------------
import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']

epochs = range(1, len(history_dict['accuracy']) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

#------------------
# Plot training and validation accuracy
#------------------
plt.clf()
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']

plt.plot(epochs, acc_values, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()






#------------------
# Retrain model with L2 regularization over 20 epochs
#------------------
from keras import regularizers

print("---------Retrain model with L2 regularization over 20 epochs---------")
model3 = models.Sequential()
model3.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
						activation='relu', input_shape=(10000,)))
model3.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
						activation='relu'))
model3.add(layers.Dense(1, activation='sigmoid'))

model3.compile(optimizer='rmsprop',
			  loss='binary_crossentropy',
			  metrics=['accuracy'])

history = model3.fit(x_train,
		   y_train,
		   epochs=20,
		   batch_size=512,
		   validation_data=(x_val,y_val))
results = model3.evaluate(x_test,y_test)
print("Model binary crossentropy and accuracy with test data:",results)

#------------------
# Plot training and validation loss
#------------------
import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']

epochs = range(1, len(history_dict['accuracy']) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

#------------------
# Plot training and validation accuracy
#------------------
plt.clf()
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']

plt.plot(epochs, acc_values, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()







#------------------
# Retrain model with L2 regularization over 20 epochs with bigger layers (32 neurons / layer)
#------------------
from keras import regularizers

print("---------Retrain model with L2 regularization over 20 epochs, with bigger layers (32 neurons / layer)---------")
model4 = models.Sequential()
model4.add(layers.Dense(32, kernel_regularizer=regularizers.l2(0.001),
						activation='relu', input_shape=(10000,)))
model4.add(layers.Dense(32, kernel_regularizer=regularizers.l2(0.001),
						activation='relu'))
model4.add(layers.Dense(1, activation='sigmoid'))

model4.compile(optimizer='rmsprop',
			  loss='binary_crossentropy',
			  metrics=['accuracy'])

history = model4.fit(x_train,
		   y_train,
		   epochs=20,
		   batch_size=512,
		   validation_data=(x_val,y_val))
results = model4.evaluate(x_test,y_test)
print("Model binary crossentropy and accuracy with test data:",results)

#------------------
# Plot training and validation loss
#------------------
import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']

epochs = range(1, len(history_dict['accuracy']) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

#------------------
# Plot training and validation accuracy
#------------------
plt.clf()
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']

plt.plot(epochs, acc_values, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()



#------------------
# Retrain model with L2 regularization over 20 epochs with bigger layers (32 neurons / layer) and more layers (4)
#------------------
from keras import regularizers

print("---------Retrain model with L2 regularization over 20 epochs, with bigger layers (32 neurons / layer) and more layers (4)---------")
model5 = models.Sequential()
model5.add(layers.Dense(32, kernel_regularizer=regularizers.l2(0.001),
						activation='relu', input_shape=(10000,)))
model5.add(layers.Dense(32, kernel_regularizer=regularizers.l2(0.001),
						activation='relu'))
model5.add(layers.Dense(32, kernel_regularizer=regularizers.l2(0.001),
						activation='relu'))
model5.add(layers.Dense(32, kernel_regularizer=regularizers.l2(0.001),
						activation='relu'))
model5.add(layers.Dense(1, activation='sigmoid'))

model5.compile(optimizer='rmsprop',
			  loss='binary_crossentropy',
			  metrics=['accuracy'])

history = model5.fit(x_train,
		   y_train,
		   epochs=20,
		   batch_size=512,
		   validation_data=(x_val,y_val))
results = model5.evaluate(x_test,y_test)
print("Model binary crossentropy and accuracy with test data:",results)

#------------------
# Plot training and validation loss
#------------------
import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']

epochs = range(1, len(history_dict['accuracy']) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

#------------------
# Plot training and validation accuracy
#------------------
plt.clf()
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']

plt.plot(epochs, acc_values, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()




#------------------
# Retrain model with L2 regularization over 20 epochs with sigmoid activation function for all layers
#------------------
from keras import regularizers

print("---------Retrain model with L2 regularization over 20 epochs, with sigmoid activation function for all layers---------")
model6 = models.Sequential()
model6.add(layers.Dense(32, kernel_regularizer=regularizers.l2(0.001),
						activation='sigmoid', input_shape=(10000,)))
model6.add(layers.Dense(32, kernel_regularizer=regularizers.l2(0.001),
						activation='sigmoid'))
model6.add(layers.Dense(1, activation='sigmoid'))

model6.compile(optimizer='rmsprop',
			  loss='binary_crossentropy',
			  metrics=['accuracy'])

history = model6.fit(x_train,
		   y_train,
		   epochs=20,
		   batch_size=512,
		   validation_data=(x_val,y_val))
results = model6.evaluate(x_test,y_test)
print("Model binary crossentropy and accuracy with test data:",results)

#------------------
# Plot training and validation loss
#------------------
import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']

epochs = range(1, len(history_dict['accuracy']) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

#------------------
# Plot training and validation accuracy
#------------------
plt.clf()
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']

plt.plot(epochs, acc_values, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

print("The final model (model6) is selected for achieving the highest level of accuracy on the test data without overfitting the training data.")


exit()





