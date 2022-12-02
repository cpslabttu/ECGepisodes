import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import SGD
import seaborn as sns
#%matplotlib inline
import warnings
import mprof
# program to compute the time
# of execution of any python code
from datetime import datetime
from sklearn import svm
warnings.filterwarnings('ignore')
'''
dataset_train = pd.read_csv('C:/Users/as097/Desktop/PVC/tsfelWRM_multi.csv')
#print(dataset_train.head())
#open'ı kullanıcaz sadece:
#open'ı kullanıcaz sadece:
train = dataset_train.loc[:, ['Label']].values #array'e çevirdik

#print(train)

from sklearn.preprocessing import MinMaxScaler #bununla, 0-1 arasına scale ettik
scaler = MinMaxScaler(feature_range = (0, 1))
train_scaled = scaler.fit_transform(train)
print(train_scaled)


# ilk 1-50 yi alıp X_train'e, 51. data point'i de y_train'e,
# 2-51'i alıp X_train'e, 52'yi y_train'e ...olacak şekilde data frame i oluşturuyoruz:
X_train = []
y_train = []
timesteps = 50

for i in range(timesteps, 1250):
    X_train.append(train_scaled[i - timesteps:i, 0])
    y_train.append(train_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
#Reshaping:
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
#import libraries and packages:
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Dropout

#Initialize RNN:
regressor = Sequential()

#Adding the first RNN layer and some Dropout regularization
regressor.add(SimpleRNN(units = 50, activation='tanh', return_sequences=True, input_shape= (X_train.shape[1],1)))
regressor.add(Dropout(0.2))

#Adding the second RNN layer and some Dropout regularization
regressor.add(SimpleRNN(units = 50, activation='tanh', return_sequences=True))
regressor.add(Dropout(0.2))

#Adding the third RNN layer and some Dropout regularization
regressor.add(SimpleRNN(units = 50, activation='tanh', return_sequences=True))
regressor.add(Dropout(0.2))

#Adding the fourth RNN layer and some Dropout regularization
regressor.add(SimpleRNN(units = 50))
regressor.add(Dropout(0.2))

#Adding the output layer
regressor.add(Dense(units = 1))

#Compile the RNN
regressor.compile(optimizer='adam', loss='binary_crossentropy')

#Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs=10, batch_size=32)


dataset_test = pd.read_csv('C:/Users/as097/Desktop/PVC/tsfelWRM_multi.csv')
dataset_test.head()
real_stock_price = dataset_test.loc[:, ['Label']].values
print(real_stock_price)
#Getting the predicted stock price
dataset_total = pd.concat((dataset_train['Label'], dataset_test['Label']), axis=0)
inputs = dataset_total[len(dataset_total)-len(dataset_test) - timesteps:].values.reshape(-1,1)
inputs = scaler.transform(inputs) #minmax scaler

print(inputs)
X_test = []
for i in range(timesteps, 70):
    X_test.append(inputs[i-timesteps:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
#inverse_transform ile, scale edildikten sonra predict edilen değerleri gerçek değer aralığına çekiyoruz
plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
'''

data = pd.read_csv('C:/Users/as097/Desktop/PVC/tsfelWRM_multi.csv')
print(data.head())
dataset = data.iloc[:, 1].values
'''
plt.plot(dataset)
plt.xlabel('time')
plt.ylabel('number of passengers (in thousands)')
plt.title('Passengers')
plt.show()
'''
dataset = dataset.reshape(-1,1) #(145, ) iken (145,1)e çevirdik
dataset = dataset.astype('float32')
print(dataset.shape)
from sklearn.preprocessing import MinMaxScaler #bununla, 0-1 arasına scale ettik
scaler = MinMaxScaler(feature_range= (0,1))
dataset = scaler.fit_transform(dataset)
train_size = int(len(dataset)*0.3)
test_size = len(dataset)- train_size

train = dataset[0:train_size, :]
test = dataset[train_size:len(dataset), :]

print('train size: {}, test size: {}'.format(len(train), len(test)))
dataX = []
datay = []
timestemp = 10

for i in range(len(train) - timestemp - 1):
    a = train[i:(i + timestemp), 0]
    dataX.append(a)
    datay.append(train[i + timestemp, 0])

trainX, trainy = np.array(dataX), np.array(datay)
dataX = []
datay = []
for i in range(len(test) - timestemp - 1):
    a = test[i:(i + timestemp), 0]
    dataX.append(a)
    datay.append(test[i + timestemp, 0])

testX, testy = np.array(dataX), np.array(datay)
print(trainX.shape)

trainX = np.reshape(trainX, (trainX.shape[0],1,  trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0],1,  testX.shape[1]))
print(trainX.shape)

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# model
model = Sequential()
model.add(LSTM(10, input_shape=(1, timestemp))) # 10 lstm neuron(block)
model.add(Dense(1))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(trainX, trainy, epochs=10, batch_size=1)

#make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainy = scaler.inverse_transform([trainy])
testPredict = scaler.inverse_transform(testPredict)
testy = scaler.inverse_transform([testy])

import math
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainy[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testy[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# shifting train
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[timestemp:len(trainPredict)+timestemp, :] = trainPredict
# shifting test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(timestemp*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()