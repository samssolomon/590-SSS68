
import os
from keras.datasets import imdb
from keras import preprocessing
import numpy as np
from keras.models import Sequential 
from keras import layers
from keras.layers import Embedding, Flatten, Dense, LSTM
from keras.layers import LSTM
import matplotlib.pyplot as plt
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer


#---------------------------
#USER PARAM
#---------------------------
max_features = 10000    #DEFINES SIZE OF VOCBULARY TO USE
maxlen       = 30       #CUTOFF SENTENCES
epochs       = 10
batch_size   = 256
verbose      = 1
embed_dim    = 100      #DIMENSION OF EMBEDING SPACE (SIZE OF VECTOR FOR EACH WORD)
lr           = 0.001    #LEARNING RATE


#------------------------------------------------------#
# Load the data saved in 01-clean.py
#------------------------------------------------------#

corpus_tokenized = np.loadtxt('corpus_tokenized.txt').astype(int)
corpus_labels = np.loadtxt('corpus_labels.txt').astype(int)-1

# One hot encode the corpus labels
def to_one_hot(labels, dimension=3):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

corpus_labels = to_one_hot(corpus_labels)


# Partition data into training and test sets
rand_idx = np.random.permutation(corpus_labels.shape[0])
CUT1 = int(0.8*corpus_labels.shape[0]);
train_idx, test_idx = rand_idx[:CUT1], rand_idx[CUT1:]
x_train = corpus_tokenized[train_idx,:]; y_train = corpus_labels[train_idx]
x_test = corpus_tokenized[test_idx,:]; y_test = corpus_labels[test_idx]

# Split training set into training set and validation set
rand_indices = np.random.permutation(x_train.shape[0])
CUT2 = int(0.8*x_train.shape[0]); 
train_idx, val_idx = rand_indices[:CUT2], rand_indices[CUT2:]
x_val = x_train[val_idx,:]; y_val=y_train[val_idx]
x_train = x_train[train_idx,:]; y_train=y_train[train_idx]

print('input_train shape:', x_train.shape)
print('input_val shape:', x_val.shape)
print('input_test shape:', x_test.shape)

print('output_train shape:', y_train.shape)
print('output_val shape:', y_val.shape)
print('output_test shape:', y_test.shape)

# Save training, validation, and test sets and (one-hot encoded) labels

np.savetxt('x_train.txt',x_train)
np.savetxt('y_train.txt',y_train)
np.savetxt('x_val.txt',x_val)
np.savetxt('y_val.txt',y_val)
np.savetxt('x_test.txt',x_test)
np.savetxt('y_test.txt',y_test)



#------------------------------------------------------#
# Define model
#------------------------------------------------------#

model = Sequential() 
model.add(layers.Embedding(max_features, embed_dim, input_length=maxlen))
model.add(layers.LSTM(32,
                      dropout=0.1,
                      recurrent_dropout=0.5,
                      return_sequences=True)) 
model.add(layers.LSTM(32,activation='relu',
                      dropout=0.1,
                      recurrent_dropout=0.5)) 
model.add(layers.Dense(3, activation='softmax'))
model.compile(optimizer=RMSprop(lr=lr),
              loss='categorical_crossentropy',
              metrics=['acc']) 
model.summary()





#------------------------------------------------------#
# Train model
#------------------------------------------------------#

history = model.fit(x_train,
                    y_train, 
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(x_val,y_val),
                    verbose=verbose)



#---------------------------
# Plot
#---------------------------
def report(history,title='',I_PLOT=True):

    if(I_PLOT):
        #PLOT HISTORY
        epochs = range(1, len(history.history['loss']) + 1)
        plt.figure()
        plt.plot(epochs, history.history['loss'], 'bo', label='Training loss')
        plt.plot(epochs, history.history['acc'], 'b', label='Training acc')
        plt.plot(epochs, history.history['val_loss'], 'ro', label='Validation loss')
        plt.plot(epochs, history.history['val_acc'], 'r', label='Validation acc')

        plt.title(title)
        plt.legend()
        # plt.show()

        plt.savefig('HISTORY-'+title+'.png')   # save the figure to file
        plt.close()


report(history,title="LSTM")


#---------------------------
# Save model
#---------------------------

model.save('final_LSTM_model.h5')


