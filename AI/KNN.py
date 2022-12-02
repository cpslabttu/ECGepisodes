#%matplotlib inline

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix , classification_report
from sklearn.model_selection import train_test_split

# Bagged Trees Regressor
from sklearn.ensemble import BaggingRegressor
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("C:/Users/as097/Desktop/PVC/tsfelWRM_multi.csv")
#print(df.sample(5))
#print(df.Label.value_counts())


df.drop('0_ECDF Percentile Count_0',axis='columns',inplace=True)
#print(df.dtypes)

X = df.drop('Label1',axis='columns')
y = testLabels = df.Label1.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=15, stratify=y)


def KNN(X_train, y_train, X_test, y_test):
    # training a KNN classifier
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train)

    # accuracy on X_test
    accuracy = knn.score(X_test, y_test)
    print(accuracy)
    # creating a confusion matrix
    y_preds = knn.predict(X_test)

    print("Classification Report: \n", classification_report(y_test, y_preds))
    return y_preds

y_preds = KNN(X_train, y_train, X_test, y_test)

cm = confusion_matrix(y_test, y_preds)
print(cm)