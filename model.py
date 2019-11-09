import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from keras import optimizers
import keras
from matplotlib import pyplot as plt
from IPython.display import clear_output

data = pd.read_csv('paddy.csv')
data = np.asarray(data)


# print(data.shape)
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
X = data[:, 1:7]
Y = data[:, 7]
# print(X[:,0])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1111)
# print(X_train.shape)

#################### Regression model ######################
# reg = LinearRegression().fit(X_train, Y_train)
#
# params =  reg.get_params()
#
# score = reg.score(X_train,Y_train)
# print(score)
# print(params)

# define the keras model
model = Sequential()
model.add(Dense(20, input_dim=6, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(10, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(5, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(2, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(1, activation='relu'))

opt = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)

model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])

model.fit(x=X_train, y=Y_train, batch_size=800, epochs=100, validation_data=(X_val, Y_val),
          callbacks=[plot_losses], verbose=1)

model.fit(x=X_train, y=Y_train, batch_size=800, epochs=100, validation_data=(X_val, Y_val),
          callbacks=[plot_losses], verbose=1)

model.fit(x=X_train, y=Y_train, batch_size=800, epochs=100, validation_data=(X_val, Y_val),
          callbacks=[plot_losses], verbose=1)

model.save('Paddy.h5')

y_pred=model.predict(X_test)

for i in range(y_pred.shape[0]):
    print('Prediction - ',y_pred[i],'Actual - ',Y_test[i])


