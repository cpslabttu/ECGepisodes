from keras import Sequential
from keras.layers import LSTM, Dropout, Dense
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings('ignore')
'''
df = pd.read_csv("C:/Users/as097/Desktop/PVC/tsfelWRM_multi.csv")
dfset = df.iloc[:, 1].values
#print(df.sample(5))
#print(df.Label.value_counts())
df.drop('0_ECDF Percentile Count_0',axis='columns',inplace=True)
#print(df.dtypes)
X = df.drop('Label',axis='columns')
X = dfset.reshape(-1,1)

y = testLabels = df.Label1.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=15, stratify=y)
#X_train = np.random.random((1000,20))
#y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000,1)), num_classes = 10)
#X_test = np.random.random((1000,20))
#y_test = keras.utils.to_categorical(np.random.randint(10, size=(1000,1)), num_classes = 10)
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
scaler = MinMaxScaler(feature_range= (0,1,2,3))
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

#mnist = tf.keras.datasets.mnist  # mnist is a dataset of 28x28 images of handwritten digits and their labels
#(x_train, y_train),(x_test, y_test) = mnist.load_data()  # unpacks images to x_train/x_test and labels to y_train/y_test
#x_train = X_train/255.0
#x_test = X_test/255.0

#print(X_train.shape)
#print(X_train[0].shape)
model = Sequential()
model.add(LSTM(128, input_shape=(1, timestemp), activation='relu', return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(128, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(4, activation='softmax'))
opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'],
)
Bi = model.fit(trainX,
          trainy,
          epochs=3,
          validation_data=(testX, testy))
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

#print('\n', classification_report(trainX, trainy))
