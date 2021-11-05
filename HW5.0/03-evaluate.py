
import numpy as np
from keras.models import load_model

#---------------------------
# Load data
#---------------------------

x_train = np.loadtxt('x_train.txt').astype(int)
y_train = np.loadtxt('y_train.txt').astype(int)

x_val = np.loadtxt('x_val.txt').astype(int)
y_val = np.loadtxt('y_val.txt').astype(int)

x_test = np.loadtxt('x_test.txt').astype(int)
y_test = np.loadtxt('y_test.txt').astype(int)

#---------------------------
# Load model
#---------------------------

model = load_model('final_LSTM_model.h5')
model.summary()

#---------------------------
# Evaluate model on training, validation, and test sets
#---------------------------

results_train = model.evaluate(x_train,y_train)
results_val = model.evaluate(x_val,y_val)
results_test = model.evaluate(x_test,y_test)

print("Model category crossentropy and accuracy with training data:", results_train)
print("Model category crossentropy and accuracy with validation data:", results_val)
print("Model category crossentropy and accuracy with test data:", results_test)


