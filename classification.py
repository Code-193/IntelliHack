import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras import regularizers
from keras import optimizers
import keras
from matplotlib import pyplot as plt
from IPython.display import clear_output
import pickle

data = pd.read_csv('crop.csv')
data = np.asarray(data)
#print(data)
class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show();
        
plot_losses = PlotLosses()
X = data[:,1:8]
Y = data[:,8]

#print(X[:,0])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1111)
#print(Y_test)
num_classes = 3
Y_train = keras.utils.to_categorical(Y_train, num_classes) 
Y_val = keras.utils.to_categorical(Y_val, num_classes) 
#Y_test = keras.utils.to_categorical(Y_test, num_classes) 

#print(Y_test)
# define the keras model
model = Sequential()
model.add(Dense(10 , input_dim=7, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
#model.add(Dropout(0.2))
model.add(Dense(6, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
#model.add(Dropout(0.2))
#model.add(Dense(2, activation='sigmoid',kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(3, activation='softmax'))

opt = keras.optimizers.rmsprop(lr=0.01, decay=1e-6)
 
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
model.fit(x=X_train, y=Y_train, batch_size=800, epochs=100, validation_data=(X_val, Y_val),
          callbacks=[plot_losses],verbose=1)

model.fit(x=X_train, y=Y_train, batch_size=800, epochs=100, validation_data=(X_val, Y_val),
          callbacks=[plot_losses],verbose=1)

model.fit(x=X_train, y=Y_train, batch_size=800, epochs=100, validation_data=(X_val, Y_val),
          callbacks=[plot_losses],verbose=1)
model.fit(x=X_train, y=Y_train, batch_size=800, epochs=100, validation_data=(X_val, Y_val),
          callbacks=[plot_losses],verbose=1)

def accuracy(pred,Y):    
    score=np.sum(pred==Y)*100/Y.shape[0]    
    return score 
model.save('Crop.h5')
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
test_score=accuracy(y_pred,Y_test)

print(test_score)

