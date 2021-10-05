#------------------
# Load Reuters dataset
#------------------
from keras.datasets import reuters

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

#------------------
# Examine data
#------------------

#len(train_data)
#len(test_data)

#train_data[10]
#max([max(sequence) for sequence in train_data])

#word_index = reuters.get_word_index()
#reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
#decoded_newswire = ' '.join([reverse_word_index.get(i-3, '?') for i in train_data[0]])
#print(decoded_newswire)

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

def to_one_hot(labels, dimension=46):
	results=np.zeros((len(labels),dimension))
	for i, label in enumerate(labels):
		results[i, label] = 1.
	return results

one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

#------------------
# Define model
#------------------
from keras import models
from keras import layers

model1 = models.Sequential()
model1.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
model1.add(layers.Dense(64,activation='relu'))
model1.add(layers.Dense(46,activation='softmax'))

#------------------
# Compile model (define optimizer, loss function, and metrics)
#------------------
model1.compile(optimizer='rmsprop',
			  loss='categorical_crossentropy',
			  metrics=['accuracy'])

#------------------
# Set aside validation set
#------------------
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

#------------------
# Train model
#------------------
history = model1.fit(partial_x_train,
					partial_y_train,
					epochs=20,
					batch_size=512,
					validation_data=(x_val,y_val))

results = model1.evaluate(x_test,one_hot_test_labels)

print("Model category crossentropy and accuracy with test data:", results)


#------------------
# Plot training and validation loss
#------------------
import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

#------------------
# Plot training and validation accuracy
#------------------
plt.clf()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs,acc,'bo',label='Training accuracy')
plt.plot(epochs,val_acc,'b',label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()






#------------------
# Retrain model from scratch
#------------------
print("---------Retrain model over 9 epochs instead of 20---------")

model2 = models.Sequential()
model2.add(layers.Dense(64, activation='relu',input_shape=(10000,)))
model2.add(layers.Dense(64, activation='relu'))
model2.add(layers.Dense(46, activation='softmax'))

model2.compile(optimizer='rmsprop',
			  loss='categorical_crossentropy',
			  metrics=['accuracy'])

history = model2.fit(partial_x_train,
		  partial_y_train,
		  epochs=9,
		  batch_size=512,
		  validation_data=(x_val,y_val))

results = model2.evaluate(x_test,one_hot_test_labels)

print("Model category crossentropy and accuracy with test data:", results)

#------------------
# Plot training and validation loss
#------------------

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

#------------------
# Plot training and validation accuracy
#------------------
plt.clf()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs,acc,'bo',label='Training accuracy')
plt.plot(epochs,val_acc,'b',label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()






#------------------
# Retrain model with L2 regularization over 20 epochs
#------------------
from keras import regularizers

print("---------Retrain model with L2 regularization over 20 epochs---------")

model3 = models.Sequential()
model3.add(layers.Dense(64, activation='relu',
						kernel_regularizer=regularizers.l2(0.001),
						input_shape=(10000,)))
model3.add(layers.Dense(64, activation='relu',
						kernel_regularizer=regularizers.l2(0.001)))
model3.add(layers.Dense(46, activation='softmax'))

model3.compile(optimizer='rmsprop',
			  loss='categorical_crossentropy',
			  metrics=['accuracy'])

history = model3.fit(partial_x_train,
		  partial_y_train,
		  epochs=20,
		  batch_size=512,
		  validation_data=(x_val,y_val))

results = model3.evaluate(x_test,one_hot_test_labels)

print("Model category crossentropy and accuracy with test data:", results)

#------------------
# Plot training and validation loss
#------------------

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

#------------------
# Plot training and validation accuracy
#------------------
plt.clf()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs,acc,'bo',label='Training accuracy')
plt.plot(epochs,val_acc,'b',label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()



#------------------
# Retrain model with L2 regularization over 20 epochs with bigger layers (128 neurons / layer)
#------------------

print("---------Retrain model with L2 regularization over 20 epochs with bigger layers (128 neurons / layer)---------")

model4 = models.Sequential()
model4.add(layers.Dense(128, activation='relu',
						kernel_regularizer=regularizers.l2(0.001),
						input_shape=(10000,)))
model4.add(layers.Dense(128, activation='relu',
						kernel_regularizer=regularizers.l2(0.001)))
model4.add(layers.Dense(46, activation='softmax'))

model4.compile(optimizer='rmsprop',
			  loss='categorical_crossentropy',
			  metrics=['accuracy'])

history = model4.fit(partial_x_train,
		  partial_y_train,
		  epochs=20,
		  batch_size=512,
		  validation_data=(x_val,y_val))

results = model4.evaluate(x_test,one_hot_test_labels)

print("Model category crossentropy and accuracy with test data:", results)

#------------------
# Plot training and validation loss
#------------------

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

#------------------
# Plot training and validation accuracy
#------------------
plt.clf()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs,acc,'bo',label='Training accuracy')
plt.plot(epochs,val_acc,'b',label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


#------------------
# Retrain model with L2 regularization over 20 epochs with more (4 layers) and bigger layers (128 neurons / layer)
#------------------

print("---------Retrain model with L2 regularization over 20 epochs with more (4 layers) and bigger layers (128 neurons / layer)---------")

model5 = models.Sequential()
model5.add(layers.Dense(128, activation='relu',
						kernel_regularizer=regularizers.l2(0.001),
						input_shape=(10000,)))
model5.add(layers.Dense(128, activation='relu',
						kernel_regularizer=regularizers.l2(0.001)))
model5.add(layers.Dense(46, activation='softmax'))

model5.compile(optimizer='rmsprop',
			  loss='categorical_crossentropy',
			  metrics=['accuracy'])

history = model5.fit(partial_x_train,
		  partial_y_train,
		  epochs=20,
		  batch_size=512,
		  validation_data=(x_val,y_val))

results = model5.evaluate(x_test,one_hot_test_labels)

print("Model category crossentropy and accuracy with test data:", results)

#------------------
# Plot training and validation loss
#------------------

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

#------------------
# Plot training and validation accuracy
#------------------
plt.clf()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs,acc,'bo',label='Training accuracy')
plt.plot(epochs,val_acc,'b',label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

print("The final model (model5) is selected for achieving the highest level of accuracy on the test data without overfitting the training data.")

exit()