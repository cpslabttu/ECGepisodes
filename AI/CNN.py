'''
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPool2D

seg_train = 'C:/Users/as097/Desktop/PVC/tsfelWRM_multi.csv/'
seg_test = 'C:/Users/as097/Desktop/PVC/tsfelWRM_multi1.csv/'
seg_pred = 'C:/Users/as097/Desktop/PVC/tsfelWRM_multi2.csv/'
#train = ImageDataGenerator(rescale =1. / 255)
#validation = ImageDataGenerator(rescale =1. / 255)
#test = ImageDataGenerator(rescale =1. / 255)
train_set = seg_train,
                                         classes = ["0",
                                          "1",
                                          "2",
                                          "3"],
                                         class_mode = 'categorical'
validation_set = seg_pred,
                                        target_size = (670, 390),
                                        batch_size = 32,
                                        classes=["0",
                                                 "1",
                                                "2",
                                                "3"],
                                        class_mode = 'categorical'

test_set = seg_test,
                                        target_size = (670, 390),
                                        batch_size = 32,
                                        classes=["0",
                                                "1",
                                                "2",
                                                "3"],
                                        class_mode = 'categorical'


X_train,y_train = train_set.next()
X_test,y_test = test_set.next()
model = tf.keras.models.Sequential([
tf.keras.layers.Conv2D(16,(3,3),activation= 'relu', input_shape=(670, 390,3)),
tf.keras.layers.MaxPool2D(2,2),
tf.keras.layers.Conv2D(32,(3,3),activation= 'relu'),
tf.keras.layers.MaxPool2D(2,2),
tf.keras.layers.Conv2D(64,(3,3),activation= 'relu'),
tf.keras.layers.MaxPool2D(2,2),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(512, activation='relu'),
tf.keras.layers.Dense(4,activation ='softmax')
])
model.compile(optimizer="adam", loss= "categorical_crossentropy", metrics=["accuracy"])
model.summary()
model_fit = model.fit(train_set, epochs = 12, verbose=1, batch_size=32, validation_data = validation_set)
score1 = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', score1[0])
print('Test accuracy:', score1[1])
predictions=model.predict(X_test[:10])
print(predictions)
'''

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import warnings
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn import svm

warnings.filterwarnings('ignore')
import pandas as pd
from tensorflow import keras
import tensorflow as tf

# generate data

import numpy as np
df = pd.read_csv("C:/Users/as097/Desktop/PVC/tsfelWRM_multi.csv")
df.drop('0_ECDF Percentile Count_0',axis='columns',inplace=True)
#print(df.dtypes)

X = df.drop('Label1',axis='columns')
## Multiple output
y = testLabels = df.Label1.astype(np.float32)
## Binary class
#y = testLabels = df.Label.astype(np.float32)
#print((y).head)
from sklearn.model_selection import train_test_split
#X_train = np.random.random((1000,20))
#y_train = keras.utils.to_categorical(np.random.randint(10,size=(1000,1)), num_classes=10)
#X_test =  np.random.random((100,20))
#y_test = keras.utils.to_categorical(np.random.randint(10,size=(100,1)), num_classes=10)
#print(X_train[50:51])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3,train_size=0.7, random_state=15, stratify=y)
#print(y_train.value_counts())
#print(y.value_counts())
#print(y.value_counts())
#print(len(X_train.columns))

'''
################################################## SVM classifier ###########################################
from sklearn.metrics import confusion_matrix , classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


model = svm.SVC(kernel='linear', C=0.01)
y_pred = model.fit(X_train, y_train).predict(X_test)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


from sklearn.metrics import classification_report
# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Classification report
targets = ['Class-0', 'Class-1', 'Class-2', 'Class-3']
print('\n', classification_report(y_test, y_pred, target_names=targets))
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# after creating the confusion matrix, for better understaning plot the cm.
import seaborn as sn
#plt.figure(figsize = (10,7))
#sn.heatmap(cm, annot=True)
#plt.xlabel('Predicted')
#plt.ylabel('Truth')
#plt.show()

################################################## SVM ###############################################
'''
'''
################################################## Decision Tree ###############################################

###### Decision Tree Classifier with criterion gini index

# import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
# instantiate the DecisionTreeClassifier model with criterion gini index
clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=15)

# fit the model
clf_gini.fit(X_train, y_train)

#Predict the Test set results with criterion gini index
y_pred_gini = clf_gini.predict(X_test)
from sklearn.metrics import accuracy_score

print('Model accuracy score with criterion gini index: {0:0.4f}'. format(accuracy_score(y_test, y_pred_gini)))
y_pred_train_gini = clf_gini.predict(X_train)

print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train_gini)))
# print the scores on training and test set
print('Training set score: {:.4f}'.format(clf_gini.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(clf_gini.score(X_test, y_test)))

##### Decision Tree Classifier with with criterion entropy

# instantiate the DecisionTreeClassifier model with criterion entropy
clf_en = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=15)

# fit the model
clf_en.fit(X_train, y_train)
y_pred_en = clf_en.predict(X_train)
from sklearn.metrics import accuracy_score
print('Model accuracy score with criterion entropy: {0:0.4f}'. format(accuracy_score(y_train, y_pred_en)))

y_pred_train_en = clf_en.predict(X_train)
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train_en)))

# print the scores on training and test set
print('Training set score: {:.4f}'.format(clf_en.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(clf_en.score(X_train, y_train)))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train, y_pred_en)
print('Confusion matrix\n\n', cm)

from sklearn.metrics import classification_report
print(classification_report(y_train, y_pred_en))

################################################## Decision Tree ###############################################
'''
#from memory_profiler import profile
################################################## CNN ###############################################
@profile
def CNN(X_train, y_train, X_test, y_test, loss, weights):
# define baseline model

	# create model
    model = Sequential()
    model.add(Dense(64, input_dim=159, activation='relu'))
    model.add(Dense(4, activation='softmax'))
        # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Compile model

    model.fit (X_train,y_train,epochs=100, batch_size=15, verbose=0)

    score = model.evaluate(X_train,y_train,batch_size=128,)

    y_preds = model.predict(X_train)
    y_preds = np.argmax(y_preds, axis=1)
    print("Classification Report: \n", classification_report(y_train, y_preds))
    return y_preds
y_preds = CNN(X_train, y_train, X_test, y_test, 'sparse_categorical_crossentropy', -1)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train, y_preds)
print("Cconfusion Matrix: \n",cm)

'''
if __name__ == '__main__':
    ANN(X_train, y_train, X_test, y_test)
'''
import time
import multiprocessing as mp
import psutil
import numpy as np
from requests.packages import target
import matplotlib.pyplot as plt
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

'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# Create confusion matrix
#cm = confusion_matrix(y_test, y_pred)
print("Classification Report: \n", classification_report(y_train, y_preds))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train, y_preds)
print("Cconfusion Matrix: \n",cm)
'''
################################################## CNN ###############################################

'''
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
model = keras.Sequential()
pred_X_train = model.predict(X_train)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(159,)))
#model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(4, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
model.fit(pred_X_train, y_train, epochs=100, verbose=2)
from sklearn.metrics import classification_report
# predict
pred = model.predict(pred_X_train, batch_size = 64)
pred = np.argmax(pred, axis=1)
print(classification_report(y_train, pred))
'''