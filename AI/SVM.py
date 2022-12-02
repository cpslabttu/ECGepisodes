# importing necessary libraries
from sklearn import datasets
from sklearn.metrics import confusion_matrix , classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("C:/Users/as097/Desktop/PVC/tsfelWRM_multi.csv")
#print(df.sample(5))
#print(df.Label.value_counts())


df.drop('0_ECDF Percentile Count_0',axis='columns',inplace=True)
#print(df.dtypes)

X = df.drop('Label1',axis='columns')
y = testLabels = df.Label1.astype(np.float32)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=15, stratify=y)
# dividing X, y into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
'''
def SVM(X_train, y_train, X_test, y_test):
    from sklearn.svm import SVC

    svm_model_linear = SVC(kernel='linear', C=1).fit(X_train, y_train)
    y_preds = svm_model_linear.predict(X_test)

    # model accuracy for X_test
    score = svm_model_linear.score(X_test, y_test)
    print(score)
    print("Classification Report: \n", classification_report(y_test, y_preds))
    return y_preds

y_preds = SVM(X_train, y_train, X_test, y_test)

cm = confusion_matrix(y_test, y_preds)
print(cm)
'''
# training a linear SVM classifier
from sklearn.svm import SVC

svm_model_linear = SVC(kernel='linear', C=1).fit(X_train, y_train)
svm_predictions = svm_model_linear.predict(X_test)

# model accuracy for X_test
accuracy = svm_model_linear.score(X_test, y_test)
print(accuracy)


# creating a confusion matrix
cm = confusion_matrix(y_test, svm_predictions)
print(cm)