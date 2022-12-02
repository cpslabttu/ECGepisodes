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

from requests.packages import target

warnings.filterwarnings('ignore')
# importing the library
# we initialize the variable start to
# store the starting time of execution
# of program
start = datetime.now()

df = pd.read_csv("C:/Users/as097/Desktop/PVC/tsfelWRM_multi.csv")
#print(df.sample(5))
#print(df.Label.value_counts())

df.drop('0_ECDF Percentile Count_0',axis='columns',inplace=True)
#print(df.dtypes)

X = df.drop('Label',axis='columns')
y = testLabels = df.Label.astype(np.float32)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.9, random_state=15, stratify=y)
#print(y_train.value_counts())
#print(y.value_counts())
#print(y.value_counts())
#print(len(X_train.columns))

from tensorflow_addons import losses
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix , classification_report

from memory_profiler import profile
# instantiating the decorator
@profile
# code for which memory has to
# be monitored
def ANN(X_train, y_train, X_test, y_test, loss, weights):
    model = keras.Sequential([
        keras.layers.Dense(1, input_dim=159, activation='relu'),
        keras.layers.Dense(1, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

    if weights == -1:
        model.fit(X_train, y_train, epochs=100)
    else:
        model.fit(X_train, y_train, epochs=100, class_weight=weights)

    print(model.evaluate(X_train, y_train))

    y_preds = model.predict(X_train)
    y_preds = np.round(y_preds)

    print("Classification Report: \n", classification_report(y_train, y_preds))

    return y_preds
y_preds = ANN(X_train, y_train, X_test, y_test, 'binary_crossentropy', -1)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train, y_preds)
print("Cconfusion Matrix: \n",cm)


#X = df.drop('Label1',axis='columns')
Xs = df.drop('Label1',axis='columns')
X = Xs[df['Label1'] != 1 & 2 & 3]
ys = testLabels = df.Label1.astype(np.float32)
y = ys[df['Label1'] != 1 & 2 & 3]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2,train_size=0.8, random_state=15, stratify=y)
#print(y_train.value_counts())
#print(y.value_counts())
#print(y.value_counts())
#print(len(X_train.columns))

from tensorflow_addons import losses
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix , classification_report

@profile
# code for which memory has to
# be monitored
def ANNM(X_train, y_train, X_test, y_test, loss, weights):
    model = keras.Sequential([
        keras.layers.Dense(1, input_dim=159, activation='relu'),
        keras.layers.Dense(1, activation='relu'),
        keras.layers.Dense(4, activation='softmax')

    ])

    # compile the model
    model.compile(optimizer='Adam', loss= loss, metrics=['accuracy'])

    if weights == -1:
        model.fit(X_train, y_train, epochs=100)
    else:
        model.fit(X_train, y_train, epochs=100, class_weight=weights)

    print(model.evaluate(X_train, y_train))

    y_preds = model.predict(X_train)
    y_preds = np.argmax(y_preds, axis=1)

    print("Classification Report: \n", classification_report(y_train, y_preds))
    return y_preds
y_preds = ANNM(X_train, y_train, X_test, y_test, 'sparse_categorical_crossentropy', -1)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train, y_preds)
print("Cconfusion Matrix: \n",cm)


 # now we have initialized the variable
# end to store the ending time after
# execution of program
end = datetime.now()
# difference of start and end variables
# gives the time of execution of the
# program in between
print("The time of execution of above program is :",
      str(end-start)[5:])

#if __name__ == '__main__':
    #ANN(X_train, y_train, X_test, y_test),
    #ANNM(X_train, y_train, X_test, y_test),

import time
import multiprocessing as mp
import psutil
import numpy as np
def monitor(target):
    worker_process = mp.Process(target=target)
    worker_process.start()
    p = psutil.Process(worker_process.pid)
    # log cpu usage of `worker_process` every 10 ms
    cpu_percents = []
    while worker_process.is_alive():
        cpu_percents.append(p.cpu_percent())
        time.sleep(0.01)
    worker_process.join()
    return cpu_percents

if __name__ ==  '__main__':
    cpu_percents = monitor(target)
    plt.plot(cpu_percents)
    plt.show()