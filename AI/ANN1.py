import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv("C:/Users/as097/Desktop/PVC/tsfelWRM_imbal.csv")
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


def ANN(X_train, y_train, X_test, y_test, loss, weights):
    model = keras.Sequential([
        keras.layers.Dense(159 , input_dim=159, activation='relu'),
        keras.layers.Dense(15, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

    if weights == -1:
        model.fit(X_train, y_train, epochs=10)
    else:
        model.fit(X_train, y_train, epochs=10, class_weight=weights)

    print(model.evaluate(X_test, y_test))

    y_preds = model.predict(X_test)
    y_preds = np.round(y_preds)

    print("Classification Report: \n", classification_report(y_test, y_preds))

    return y_preds
y_preds = ANN(X_train, y_train, X_test, y_test, 'binary_crossentropy', -1)

#######################################  Mitigating Skewdness of Data   ############################################
'''
#####  Method 1: Undersampling  #####

# Class count
count_class_0, count_class_1 = df.Label.value_counts()
#count_class_0, count_class_1,count_class_2, count_class_3  = df.Label1.value_counts()
# Divide by class
df_class_0 = df[df['Label'] == 0]
df_class_1 = df[df['Label'] == 1]
#df_class_0 = df[df['Label1'] == 0]
#df_class_1 = df[df['Label1'] == 1]
#df_class_2 = df[df['Label1'] == 2]
#df_class_3 = df[df['Label1'] == 3]

# Undersample 0-class and concat the DataFrames of both class
df_class_0_under = df_class_0.sample(count_class_1)
#df_class_1_under = df_class_1.sample(count_class_2)
#df_class_2_under = df_class_2.sample(count_class_3)
df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)
#df_test_under = pd.concat([df_class_0_under,df_class_1_under,df_class_2_under, df_class_1, df_class_2, df_class_3], axis=0)

#print('Random under-sampling:')
#print(df_test_under.Label.value_counts())

X = df_test_under.drop('Label',axis='columns')
y = df_test_under['Label']
#X = df_test_under.drop('Label1',axis='columns')
#y = df_test_under['Label1']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=15, stratify=y)
# Number of classes in training Data
#print(y_train.value_counts())
y_preds = ANN(X_train, y_train, X_test, y_test, 'binary_crossentropy', -1)
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_preds)
# after creating the confusion matrix, for better understaning plot the cm.
import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

'''
#####  Method2: Oversampling  #####

# Oversample 1-class and concat the DataFrames of both classes
df_class_1_over = df_class_1.sample(count_class_0, replace=True)
df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)

#print('Random over-sampling:')
#print(df_test_over.Label.value_counts())

X = df_test_over.drop('Label',axis='columns')
y = df_test_over['Label']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=15, stratify=y)
# Number of classes in training Data
#print(y_train.value_counts())

loss = keras.losses.BinaryCrossentropy()
weights = -1
y_preds = ANN(X_train, y_train, X_test, y_test, 'binary_crossentropy', -1)


#####  Method3: SMOTE  #####

X = df.drop('Label',axis='columns')
y = df['Label']
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='minority')
X_sm, y_sm = smote.fit_resample(X, y)

#print(y_sm.value_counts())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.9, random_state=15, stratify=y_sm)
# Number of classes in training Data
#print(y_train.value_counts())

y_preds = ANN(X_train, y_train, X_test, y_test, 'binary_crossentropy', -1)

#####  Method4: Use of Ensemble with undersampling  #####

# Regain Original features and labels
X = df.drop('Label',axis='columns')
y = df['Label']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=15, stratify=y)
#print(y_train.value_counts())

df3 = X_train.copy()
df3['Label'] = y_train
#print(df3.head())

df3_class0 = df3[df3.Label==0]
df3_class1 = df3[df3.Label==1]
def get_train_batch(df_majority, df_minority, start, end):
    df_train = pd.concat([df_majority[start:end], df_minority], axis=0)

    X_train = df_train.drop('Label', axis='columns')
    y_train = df_train.Label
    return X_train, y_train
X_train, y_train = get_train_batch(df3_class0, df3_class1, 0, 368)
y_pred1 = ANN(X_train, y_train, X_test, y_test, 'binary_crossentropy', -1)

X_train, y_train = get_train_batch(df3_class0, df3_class1, 368, 736)
y_pred2 = ANN(X_train, y_train, X_test, y_test, 'binary_crossentropy', -1)

X_train, y_train = get_train_batch(df3_class0, df3_class1, 736, 1104)
y_pred3 = ANN(X_train, y_train, X_test, y_test, 'binary_crossentropy', -1)

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

