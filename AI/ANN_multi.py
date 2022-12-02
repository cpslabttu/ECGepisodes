import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import SGD
import seaborn as sns
#%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv("C:/Users/as097/Desktop/PVC/tsfelWRM_multi.csv")
#print(df.sample(5))
#print(df.Label.value_counts())

df.drop('0_ECDF Percentile Count_0',axis='columns',inplace=True)
#print(df.dtypes)

X = df.drop('Label1',axis='columns')
#Xs = df.drop('Label1',axis='columns')
#X = Xs[df['Label1'] != 1 & 2 & 3]
y = testLabels = df.Label1.astype(np.float32)
#ys = testLabels = df.Label1.astype(np.float32)
#y = ys[df['Label1'] != 1 & 2 & 3]

#print(Xs)

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


def ANN(X_train, y_train, X_test, y_test, loss, weights):
    model = keras.Sequential([
        keras.layers.Dense(64, input_dim=158, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
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
y_preds = ANN(X_train, y_train, X_test, y_test, 'sparse_categorical_crossentropy', -1)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train, y_preds)
print("Cconfusion Matrix: \n",cm)

#######################################  Mitigating Skewdness of Data   ############################################

'''
#####  Method 1: Undersampling  #####

# Class count
df1 = df.groupby('Label1').apply(lambda x: x.sample(df.Label1.value_counts().min()))
count_class_0, count_class_1,count_class_2, count_class_3  = df1.Label1.value_counts()
# Divide by class

df1_class_0 = df1[df1['Label1'] == 0]
df1_class_1 = df1[df1['Label1'] == 1]
df1_class_2 = df1[df1['Label1'] == 2]
df1_class_3 = df1[df1['Label1'] == 3]

# Undersample 0-class and concat the DataFrames of both class
df1_class_0_under = df1_class_0.sample(count_class_1 )
df1_class_1_under = df1_class_0.sample(count_class_2 )
df1_class_2_under = df1_class_0.sample(count_class_3 )

df1_test_under = pd.concat([df1_class_0_under,df1_class_1_under,df1_class_2_under, df1_class_1, df1_class_2, df1_class_3], axis=0)

#print('Random under-sampling:')
print(df1_test_under.Label1.value_counts())

X = df1_test_under.drop('Label1',axis='columns')
y = df1_test_under['Label1']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=15, stratify=y)
# Number of classes in training Data
print(y_train.value_counts())


y_preds = ANN(X_train, y_train, X_test, y_test, 'sparse_categorical_crossentropy', -1)
'''

'''
#####  Method2: Oversampling  #####

count_class_0, count_class_1,count_class_2, count_class_3  = df.Label1.value_counts()
df_class_0 = df[df['Label1'] == 0]
df_class_1 = df[df['Label1'] == 1]
df_class_2 = df[df['Label1'] == 2]
df_class_3 = df[df['Label1'] == 3]

# Oversample 1-class and concat the DataFrames of both classes
df_class_1_over = df_class_1.sample(count_class_0, replace=True)
df_class_3_over = df_class_3.sample(count_class_2, replace=True)
df_test_over = pd.concat([df_class_0,df_class_2, df_class_3_over,df_class_1_over], axis=0)

#print('Random over-sampling:')
#print(df_test_over.Label.value_counts())

X = df_test_over.drop('Label1',axis='columns')
y = df_test_over['Label1']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=15, stratify=y)
# Number of classes in training Data
#print(y_train.value_counts())

loss = keras.losses.CategoricalCrossentropy()
weights = -1
y_preds = ANN(X_train, y_train, X_test, y_test, 'sparse_categorical_crossentropy', -1)
'''
'''
#####  Method3: SMOTE  #####

X = df.drop('Label1',axis='columns')
y = df['Label1']
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='minority')
X_sm, y_sm = smote.fit_resample(X, y)

#print(y_sm.value_counts())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.9, random_state=15, stratify=y_sm)
# Number of classes in training Data
#print(y_train.value_counts())

y_preds = ANN(X_train, y_train, X_test, y_test, 'sparse_categorical_crossentropy', -1)
'''
'''
#####  Method4: Use of Ensemble with undersampling  #####

# Regain Original features and labels
X = df.drop('Label1',axis='columns')
y = df['Label1']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=15, stratify=y)
#print(y_train.value_counts())

df3 = X_train.copy()
df3['Label1'] = y_train
#print(df3.head())

df3_class0 = df3[df3.Label1==0]
df3_class1 = df3[df3.Label1==1]
df3_class2 = df3[df3.Label1==2]
df3_class3 = df3[df3.Label1==3]
def get_train_batch(df_majority, df_minority1,df_minority2,df_minority3, start, end):
    df_train = pd.concat([df_majority[start:end], df_minority1,df_minority2,df_minority3], axis=0)

    X_train = df_train.drop('Label1', axis='columns')
    y_train = df_train.Label1
    return X_train, y_train
X_train, y_train = get_train_batch(df3_class0, df3_class1,df3_class2, df3_class3, 0, 400)
y_pred1 = ANN(X_train, y_train, X_test, y_test, 'sparse_categorical_crossentropy', -1)

X_train, y_train = get_train_batch(df3_class0, df3_class1,df3_class2, df3_class3, 400, 700)
y_pred2 = ANN(X_train, y_train, X_test, y_test, 'sparse_categorical_crossentropy', -1)

X_train, y_train = get_train_batch(df3_class0, df3_class1,df3_class2, df3_class3, 700, 1360)
y_pred3 = ANN(X_train, y_train, X_test, y_test, 'sparse_categorical_crossentropy', -1)

print(len(y_pred1))
y_pred_final = y_pred1.copy()
for i in range(len(y_pred1)):
    n_ones = y_pred1[i] + y_pred2[i] + y_pred3[i]
    if n_ones>1:
        y_pred_final[i] = 1
    else:
        y_pred_final[i] = 0
cl_rep = classification_report(y_test, y_pred_final)
print(cl_rep)
'''


